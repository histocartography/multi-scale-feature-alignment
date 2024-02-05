import os
import time
from typing import Dict, List, Tuple
from argparse import ArgumentParser
from copy import deepcopy

import numpy as np
import torch
from torch import optim
from torchvision.utils import make_grid
from source.da_models.abstract_da import AbstractDAModel
from source.dataloader.get_loaders import get_loaders
from source.dataloader.patch_dataset import PatchDataset

from source.networks.cond_style_gan import Discriminator, Generator
from source.networks.resnet_model import ResnetModel
from source.utils.constants import (
    DOMS,
    ITER_GAN_EPOCHS,
    MAP_DATASET_STATS,
    MAX_ITERS,
    SET_TYPES,
    STAGE_DIMS,
)
from source.utils.misc import get_n_trainable_params


class StyleGAN(object):
    """Model implementing DANN baseline, where real source data
    is substituted by synthesized data sampled from a GAN.
    """

    def __init__(
        self,
        model: ResnetModel,
        patch_dataset: PatchDataset,
        src_doms: List[str],
        tar_doms: List[str],
        device: str,
        n_classes,
        stage: int = 0,
        gan_disc_lr: float = 2e-4,
        gan_gen_lr: float = 3e-4,
        gan_epochs: int = 200,
        r1_gamma: float = 2,
        norm_stats: str = "imagenet",
        gan_batchsize: int = 32,
        not_balanced: bool = False,
        residual: bool = False,
        attention: bool = False,
        delt: float = 0.1,
        **kwargs,
    ):
        super(StyleGAN, self).__init__()
        self.tb_logger = kwargs.get("tb_logger")
        self.attention = attention
        self.residual = residual
        self.device = device
        self.feature_extractor = model
        self.patch_dataset = patch_dataset
        self.loaders_kwargs = dict(
            norm_stats=norm_stats,
            batch_size=gan_batchsize,
            device=device,
            not_balanced=not_balanced,
        )
        self.gan_train_loaders = get_loaders(
            self.patch_dataset, src_doms, tar_doms, **self.loaders_kwargs
        ).get(SET_TYPES[0], None)

        # GAN
        self.stage: int = stage
        self.n_doms = DOMS
        self.true_doms = len(src_doms) if src_doms is not None else 0
        self.discriminator: Discriminator = Discriminator(
            stage=self.stage,
            n_cls=n_classes + self.n_doms,
            device=self.device,
            mbdis=True,
            attention=attention,
            residual=residual,
        )
        self.generator: Generator = Generator(
            latent_dim=512, n_cls=n_classes + self.n_doms, device=self.device
        )
        self.gan_disc_lr: float = gan_disc_lr
        self.gan_gen_lr: float = gan_gen_lr
        self.n_cls: int = n_classes
        self.gan_epochs: int = gan_epochs
        self.gan_batchsize: int = gan_batchsize
        self.softplus = torch.nn.Softplus()
        self.l1loss = torch.nn.L1Loss(reduction="mean")
        print(
            f"Trainable parameters in D: {get_n_trainable_params(self.discriminator)}",
            flush=True,
        )
        print(
            f"Trainable parameters in G: {get_n_trainable_params(self.generator)}",
            flush=True,
        )

        # exponentially moving average (EMA) of G parameters
        self.generator_ema = deepcopy(self.generator).eval()
        self.generator_ema.requires_grad(False)

        # R1 regularization parameters
        self.r1_gamma = (
            r1_gamma
            if r1_gamma is not None
            else 0.0002 * (STAGE_DIMS[self.stage][-1] ** 2 / self.batch_size)
        )
        self.r1_interval = 16
        self.d_b1 = 0.0
        self.d_b2 = 0.99
        if self.r1_interval > 1:
            self.lazy_c = self.r1_interval / (self.r1_interval + 1)
            self.gan_disc_lr *= self.lazy_c
            self.d_b1 = self.d_b1**self.lazy_c
            self.d_b2 = self.d_b2**self.lazy_c

        # disable D augmentation
        self.aug_pipe = None

        # fixed noise vectors for visualization
        self.n_samples_per_cls = 5
        self.fixed_z = torch.randn(
            self.n_samples_per_cls * self.n_cls * self.true_doms,
            self.generator.latent_dim,
            device=self.device,
        )
        self.fixed_y = self._to_one_hot(
            torch.tensor(range(self.n_cls)).repeat(
                self.n_samples_per_cls * self.true_doms
            )
        )
        self.fixed_y = torch.cat(
            (
                self.fixed_y,
                self._to_one_hot(
                    torch.tensor(range(self.true_doms)).repeat_interleave(
                        self.n_samples_per_cls * self.n_cls
                    ),
                    self.n_doms,
                ),
            ),
            dim=1,
        )
        self.norm_stats = norm_stats
        self.step = 0
        self.delt = delt

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument(
            "--stage",
            default=0,
            type=int,
            required=False,
            help="feature stage for discriminator",
        )
        parser.add_argument(
            "--gan-disc-lr",
            default=0.0002,
            type=float,
            required=False,
            help="discriminator learning rate",
        )
        parser.add_argument(
            "--gan-gen-lr",
            default=0.0002,
            type=float,
            required=False,
            help="generator learning rate",
        )
        parser.add_argument(
            "--gan-epochs",
            default=250,
            type=int,
            required=False,
            help="number of GAN epochs",
        )
        parser.add_argument(
            "--gan-batchsize",
            default=16,
            type=int,
            required=False,
            help="number of batchsize for gan",
        )
        parser.add_argument(
            "--r1-gamma",
            type=float,
            required=False,
            help="r1 regularization coefficient (gamma)",
        )
        parser.add_argument(
            "--attention", action="store_true", help="attention in discrminator"
        )
        parser.add_argument(
            "--residual", action="store_true", help="residual in discrminator"
        )
        parser.add_argument(
            "--delt", default=0.0, type=float, required=False, help="distillation param"
        )
        return parser.parse_known_args(args)

    def train_generative(self, name, distill: bool = False) -> int:
        # set G and D as trainable, model as not trainable
        self.generator.train()
        self.generator.requires_grad(True)
        self.discriminator.train()
        self.discriminator.requires_grad(True)
        if self.source_model_dict:
            self.feature_extractor.set_state_dict(self.source_model_dict)
        self.feature_extractor.eval()
        self.feature_extractor.requires_grad(False)
        # initialize optimizers
        optimizer_disc = optim.Adam(
            self.discriminator.parameters(),
            lr=self.gan_disc_lr,
            betas=(self.d_b1, self.d_b2),
        )
        optimizer_gen = optim.Adam(
            self.generator.parameters(), lr=self.gan_gen_lr, betas=(0.0, 0.99)
        )
        # only keep random crop augmentations in the dataloader
        gan_loader = AbstractDAModel.get_loaders(
            self.gan_train_loaders, name, squeeze=True
        )
        gan_loader.dataset.init_preprocessing(
            stats_type=self.norm_stats, gan_input=True, is_tensor=False
        )
        # after 1/4 of all images have been seen, EMA ramp-up is complete (EMA beta will be at its maximum)
        ema_rampup = 500000 / (
            gan_loader.batch_size * len(gan_loader) * self.gan_epochs / 4
        )

        start_time = time.time()
        g_steps = 0
        # generate pseudolabels DB
        all_labels = self.generate_pseudolabels(name) if distill else {}

        for epoch in range(self.gan_epochs):
            g_dist_loss = torch.tensor(0.0)
            for i in range(ITER_GAN_EPOCHS):
                data, _ = gan_loader.fetch()
                real_curr_labels = (
                    data["labels"].to(self.device)
                    if not distill
                    else torch.stack([all_labels[ii] for ii in data["idx"]], dim=0).to(
                        self.device
                    )
                )
                real_curr_images = data["images"].to(self.device)
                batch_size = real_curr_images.shape[0]
                real_curr_labels = self._to_one_hot(real_curr_labels)

                # ratio of previous vs current task samples per batch
                ratio = 0.5
                if distill:
                    batch_size = int(batch_size * ratio)
                    real_curr_labels = real_curr_labels[:batch_size]
                    real_curr_images = real_curr_images[:batch_size]

                ######################
                # Update discriminator
                ######################
                # train D
                self.discriminator.requires_grad(True)
                self.discriminator.zero_grad()
                # freeze G
                self.generator.requires_grad(False)

                # sample D inputs (z used later for G update)
                imgs_r, labels_r, imgs_f, labels_f, z = self.sample_d_inputs(
                    real_curr_images, real_curr_labels, True
                )
                # if distill: real_samples|old_gan_prev_dom, fake: gan_current_dom|gan_prev_dom
                if distill:
                    imgs_r_prev, labels_r_prev, imgs_f_prev, labels_f_prev, z_prev = (
                        self.sample_d_inputs(real_curr_images, real_curr_labels, False)
                    )
                    if self.stage > 0:
                        imgs_r, imgs_f = [
                            torch.cat((imgs_r[i], imgs_r_prev[i]), dim=0)
                            for i in range(len(imgs_r))
                        ], [
                            torch.cat((imgs_f[i], imgs_f_prev[i]), dim=0)
                            for i in range(len(imgs_f))
                        ]
                    else:
                        imgs_r, imgs_f = torch.cat(
                            (imgs_r, imgs_r_prev), dim=0
                        ), torch.cat((imgs_f, imgs_f_prev), dim=0)
                    labels_r, labels_f = torch.cat(
                        (labels_r, labels_r_prev), dim=0
                    ), torch.cat((labels_f, labels_f_prev), dim=0)
                    z = torch.cat((z, z_prev), dim=0)

                # D(real)
                d_real = self.discriminator(imgs_r, labels_r)
                # D(fake)
                d_fake = self.discriminator(imgs_f, labels_f)
                # D loss (non-saturating + R1)
                d_ns_loss = torch.mean(self.softplus(d_fake) + self.softplus(-d_real))
                # (Lazy) R1 regularization (gp on reals)
                d_r1_loss = self._get_r1_loss(imgs_r, labels_r, g_steps)
                d_loss = d_ns_loss + d_r1_loss

                # Backprop and GD step
                d_loss.backward()
                optimizer_disc.step()

                ##################
                # Update generator
                ##################
                # train G
                self.generator.requires_grad(True)
                self.generator.zero_grad()
                # freeze D
                self.discriminator.requires_grad(False)

                # if GDist, explicitly sample noise maps, else sample on the fly
                noise_maps = (
                    [
                        torch.randn((z.shape[0], 1, res, res), device=self.device)
                        for res in self.generator.input_res
                    ]
                    if distill
                    else None
                )

                # generate fakes
                fake_images = self.generator(z, labels_f, noise=noise_maps)
                fake_aug_images = self.augment_non_leaking(fake_images)
                # feat. matching?
                fake_images_feat = (
                    self.feature_extractor(fake_aug_images, output_stage=self.stage)[-1]
                    if self.stage > 0
                    else fake_aug_images
                )

                # D(fake)
                d_fake_g = self.discriminator(fake_images_feat, labels_f)

                # GDist
                if distill:
                    # send half the batch through prev-G and compute GDist on it
                    noise_maps = [mp[batch_size:] for mp in noise_maps]
                    dist_imgs_gt = self.generator_old(
                        z[batch_size:], labels_f[batch_size:], noise=noise_maps
                    )
                    g_dist_loss = self.l1loss(dist_imgs_gt, fake_images[batch_size:])
                # G loss (non-saturating + GDist)
                g_loss = torch.mean(self.softplus(-d_fake_g))
                loss_gen = g_loss + self.delt * g_dist_loss

                # Backprop and GD step
                loss_gen.backward()
                optimizer_gen.step()
                g_steps += 1
                self.step += 1

                # update EMA copy
                with torch.autograd.profiler.record_function("G_ema"):
                    # linearly ramp up beta from 0 to 0.9999 (once 1/4 of images have been seen)
                    ema_nimg = min(500000, gan_loader.batch_size * g_steps * ema_rampup)
                    ema_beta = 0.5 ** (gan_loader.batch_size / max(ema_nimg, 1e-8))
                    # lin. interpolate and update
                    for p_ema, p in zip(
                        self.generator_ema.parameters(), self.generator.parameters()
                    ):
                        p_ema.copy_(p.lerp(p_ema, ema_beta))
                    # copy buffers
                    for b_ema, b in zip(
                        self.generator_ema.buffers(), self.generator.buffers()
                    ):
                        b_ema.copy_(b)

                # log scalars
                if g_steps % 50 == 0:
                    metric_dict = {
                        "L_G": g_loss.item(),
                        "L_D": d_loss.item(),
                        "L_D_GAN": d_ns_loss.item(),
                        "D_real": torch.mean(d_real).item(),
                        "D_fake": torch.mean(d_fake).item(),
                        "L_dist_G": g_dist_loss.item(),
                    }
                    self.tb_logger.run(
                        func_name="log_scalars", metric_dict=metric_dict, step=self.step
                    )

                # log generated images
                if g_steps % 1000 == 0:
                    duration = time.time() - start_time
                    print(
                        f"|| Time for 1K steps = {duration/60:.2f} minutes (epoch {epoch+1:3d}/{self.gan_epochs:3d}) ||",
                        flush=True,
                    )
                    self._log_gen_samples()
                    start_time = time.time()

                # save checkpoints
                if g_steps >= 150000 and g_steps % 10000 == 0:
                    k_steps = int(g_steps / 1000)
                    print(f"Saving checkpoints for step {k_steps}K...", flush=True)
                    torch.save(
                        self.generator.state_dict(),
                        os.path.join(
                            self.tb_logger.log_dir, f"{k_steps}K_generator.pt"
                        ),
                    )
                    torch.save(
                        self.generator_ema.state_dict(),
                        os.path.join(
                            self.tb_logger.log_dir, f"{k_steps}K_generator_ema.pt"
                        ),
                    )
                    torch.save(
                        self.discriminator.state_dict(),
                        os.path.join(
                            self.tb_logger.log_dir, f"{k_steps}K_discriminator.pt"
                        ),
                    )
                    torch.save(
                        {
                            "optimizer_gen": optimizer_gen.state_dict(),
                            "optimizer_disc": optimizer_disc.state_dict(),
                        },
                        os.path.join(
                            self.tb_logger.log_dir, f"{k_steps}K_optimizers.pt"
                        ),
                    )

        # reset the model to trainable and GAN to eval mode
        self.feature_extractor.train()
        self.feature_extractor.requires_grad(True)

        self.discriminator.eval()
        self.discriminator.requires_grad(False)
        self.generator.eval()
        self.generator.requires_grad(False)

    def sample_d_inputs(
        self, imgs_r: torch.Tensor, class_labels_r: torch.Tensor, curr: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample inputs for the discriminator"""
        # sample fake inputs
        # if curr => current domain real vs current generator on current domain as fake
        # else => old generator on old domain as real, current generator on old domain as fake
        if curr:
            dom_labs = self._to_one_hot(
                torch.tensor(self.true_doms - 1, device=self.device).repeat(
                    class_labels_r.shape[0]
                ),
                self.n_doms,
            )
            labels_r = torch.cat((class_labels_r, dom_labs), dim=1)
            labels_f = torch.cat((class_labels_r, dom_labs), dim=1)
            z = torch.randn(
                imgs_r.shape[0], self.generator.latent_dim, device=self.device
            )
            imgs_f = self.generator(z, labels_f).detach()
        else:
            dom_labs = self._to_one_hot(
                torch.randint(
                    0,
                    self.true_doms - 1,
                    (class_labels_r.shape[0],),
                    device=self.device,
                ),
                self.n_doms,
            )
            labels_r = torch.cat((class_labels_r, dom_labs), dim=1)
            labels_f = torch.cat((class_labels_r, dom_labs), dim=1)
            z = torch.randn(
                imgs_r.shape[0], self.generator.latent_dim, device=self.device
            )
            imgs_f = self.generator(z, labels_f).detach()
            imgs_r = self.generator_old(z, labels_f).detach()

        imgs_r = self.augment_non_leaking(imgs_r)
        imgs_f = self.augment_non_leaking(imgs_f)
        # if features are matched, pass through the model first
        if self.stage > 0:
            imgs_r = self.feature_extractor(imgs_r, output_stage=self.stage)[-1]
            imgs_f = self.feature_extractor(imgs_f, output_stage=self.stage)[-1]
        # return real & fake images, and sampled random noise
        return imgs_r, labels_r, imgs_f, labels_f, z

    def _get_r1_loss(
        self,
        real_img: torch.Tensor,
        real_labels: torch.Tensor,
        step: int,
        training_gan: bool = True,
    ) -> torch.Tensor:
        """(Lazy) R1 regularization"""
        # if lazy, don't compute for every mini-batch
        if step % self.r1_interval == 0:
            if isinstance(real_img, list):
                real_img = [img.detach().requires_grad_(True) for img in real_img]
                real_logits = (
                    self.discriminator(real_img, real_labels)
                    if training_gan
                    else self.model_target.discriminator(real_img, real_labels)
                )
            else:
                real_img = real_img.detach().requires_grad_(True)
                real_logits = (
                    self.discriminator(real_img, real_labels)
                    if training_gan
                    else self.model_target.discriminator(real_img, real_labels)
                )
                real_img = [real_img]

            r1_grad = torch.autograd.grad(
                outputs=[real_logits.sum()],
                inputs=real_img,
                create_graph=True,
                only_inputs=True,
            )[0]
            # compute r1 loss
            r1_penalty = r1_grad.square().sum(dim=[1, 2, 3])
            r1_loss = r1_penalty.mean() * (self.r1_gamma / 2) * self.lazy_c

            # log r1 loss
            if self.r1_interval > 1 or (self.r1_interval == 1 and step % 20 == 0):
                metric_dict = {"d_r1": r1_loss.item()}
                self.tb_logger.run(
                    func_name="log_scalars", metric_dict=metric_dict, step=step
                )
        else:
            r1_loss = 0.0

        return r1_loss

    def _log_gen_samples(self):
        """Log grid of generated samples with tensorboard"""
        # generate images
        with torch.no_grad():
            imgs = self.generator_ema(self.fixed_z, self.fixed_y)
            n_imgs = len(imgs)

            # de-normalize images and map back to range [0, 1] for plotting
            mean = torch.tensor(
                n_imgs * [MAP_DATASET_STATS[self.norm_stats]["mean"]],
                device=self.device,
            ).view(n_imgs, 3, 1, 1)
            std = torch.tensor(
                n_imgs * [MAP_DATASET_STATS[self.norm_stats]["std"]], device=self.device
            ).view(n_imgs, 3, 1, 1)
            imgs = torch.clip(imgs * std + mean, 0.0, 1.0)

            # log grid of generated images
            imgs = make_grid(imgs, nrow=self.n_cls)
            self.tb_logger.run(
                func_name="add_image",
                tag="gen_images",
                img_tensor=imgs,
                global_step=self.step,
            )

    def _to_one_hot(self, labels: torch.Tensor, max_labels: int = None) -> torch.Tensor:
        """Convert label tensor to one-hot format"""
        n_samples = len(labels)
        max_labels = max_labels if max_labels is not None else self.n_cls
        one_hot_encoding = np.zeros((n_samples, max_labels), dtype=np.float32)
        for i_label, label in enumerate(labels):
            one_hot_encoding[i_label, label] = 1.0
        return torch.tensor(one_hot_encoding, dtype=torch.float32, device=self.device)

    def make_data_loader(self, n_imgs: int, augment_sampled_imgs, transform):
        self.n_imgs = n_imgs
        self.augment_sampled_imgs = augment_sampled_imgs
        self.apply_fun = transform
        return self

    def fetch(self) -> Tuple[Dict[str, torch.Tensor], bool]:
        with torch.no_grad():
            labels_r = torch.randint(0, self.n_cls, (self.n_imgs,), device=self.device)
            domains = self._to_one_hot(
                torch.randint(0, self.true_doms, (self.n_imgs,), device=self.device),
                self.n_doms,
            )
            labels_r_ohe = torch.cat((self._to_one_hot(labels_r), domains), dim=1)
            z = torch.randn(
                self.n_imgs, self.generator_ema.latent_dim, device=self.device
            )
            labels_r_ohe = (
                self.apply_fun(labels_r_ohe)
                if self.apply_fun is not None
                else labels_r_ohe
            )
            imgs = self.generator_ema(z, labels_r_ohe)
            # undo normalization and get a 0-255 images for augmentation
            mean = torch.tensor(
                self.n_imgs * [MAP_DATASET_STATS[self.norm_stats]["mean"]],
                device=self.device,
            ).view(self.n_imgs, 3, 1, 1)
            std = torch.tensor(
                self.n_imgs * [MAP_DATASET_STATS[self.norm_stats]["std"]],
                device=self.device,
            ).view(self.n_imgs, 3, 1, 1)
            imgs = torch.clip(imgs * std + mean, 0.0, 1.0)
            return {
                "images": self.augment_sampled_imgs((255 * imgs).to(torch.uint8)),
                "labels": labels_r,
                "domains": domains,
            }, True

    def augment_non_leaking(self, img):
        if self.aug_pipe is not None:
            # augment
            img = self.aug_pipe(img)
        return img

    def update_generative(self, tar_dom):
        self.generator_old = deepcopy(self.generator_ema).eval()
        self.generator_old.requires_grad(False)
        self.true_doms = self.true_doms + 1
        self.fixed_z = torch.randn(
            self.n_samples_per_cls * self.n_cls,
            self.generator.latent_dim,
            device=self.device,
        ).repeat((self.true_doms, 1))
        self.fixed_y = self._to_one_hot(
            torch.tensor(range(self.n_cls)).repeat(
                self.n_samples_per_cls * self.true_doms
            )
        )
        self.fixed_y = torch.cat(
            (
                self.fixed_y,
                self._to_one_hot(
                    torch.tensor(range(self.true_doms)).repeat_interleave(
                        self.n_samples_per_cls * self.n_cls
                    ),
                    self.n_doms,
                ),
            ),
            dim=1,
        )
        if hasattr(self, "model"):
            self.source_model_dict = self.model.get_copy()
        self.train_generative(tar_dom, True)

    def load_model(self, path):
        self.source_model_dict = {}
        if os.path.exists(path):
            self.source_model_dict = torch.load(path)

        return os.path.exists(path)

    def load_gan(self, path: str, n_iter: int) -> bool:
        if os.path.exists(path):
            gen_fn = list(
                filter(
                    lambda x: str(n_iter) in x and "generator_ema" in x,
                    os.listdir(path),
                )
            )
            assert (
                len(gen_fn) > 0
            ), f"\n\t-> Look at n_iter, {n_iter} does not exist\n\n"
            gen_fn = gen_fn[0]
            disc_fn = list(
                filter(
                    lambda x: str(n_iter) in x and "discriminator" in x,
                    os.listdir(path),
                )
            )[0]

            self.generator_ema_dict = torch.load(os.path.join(path, gen_fn))
            self.discriminator_dict = torch.load(os.path.join(path, disc_fn))

            self.generator_ema.load_state_dict(deepcopy(self.generator_ema_dict))
            self.generator.load_state_dict(deepcopy(self.generator_ema_dict))
            self.discriminator.load_state_dict(deepcopy(self.discriminator_dict))
            print("LOADED")

        return os.path.exists(path)

    def generate_pseudolabels(self, name: str):
        tmp_labels = {}
        tmp_loaders_kwargs = dict(self.loaders_kwargs)
        tmp_loaders_kwargs["not_balanced"] = True
        tmp_gan_loader = AbstractDAModel.get_loaders(
            get_loaders(self.patch_dataset, [], [name], **tmp_loaders_kwargs).get(
                SET_TYPES[0], None
            ),
            name,
            squeeze=True,
        )
        tmp_gan_loader.dataset.init_preprocessing(
            stats_type=self.norm_stats, gan_input=True, is_tensor=False
        )
        for _ in range(MAX_ITERS):
            if set(tmp_labels.keys()) == set(tmp_gan_loader.dataset.keys_dataset):
                break
            data, _ = tmp_gan_loader.fetch()
            logits = self.feature_extractor(data["images"].to(self.device))[1]
            preds = torch.argmax(logits, dim=1)
            tmp_labels.update({data["idx"][j]: pred for j, pred in enumerate(preds)})
        return tmp_labels

    # Sequence of methods to simulate the DAModels
    def train_target(self, name: str, name_source: str):
        pass

    def train_source(self, name: str, warmup=True):
        pass

    def test_source(self, name):
        pass

    def test_target(self, name, name_source):
        pass

    def test_final(self, name_source):
        pass

    def transform_model(self):
        pass

    def adapt_sources(self, name):
        pass

    def init(self):
        pass

    def save_model(self, path, sequence):
        pass
