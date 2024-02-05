import os
from typing import List
from argparse import Namespace
from copy import deepcopy
import torch

from source.da_models.ours_adda import DAModel as Ours
from source.dataloader.abstract_dataset import get_transform
from source.dataloader.patch_dataset import PatchDataset
from source.loggers.exp_logger import ExperimentLogger
from source.networks.ours_adda_model import OursAddaModel
from source.networks.resnet_model import ResnetModel
from source.utils.constants import SET_TYPES
from source.da_models.generative import StyleGAN


class DAModel(Ours, StyleGAN):
    def __init__(
        self,
        model: ResnetModel,
        patch_dataset: PatchDataset,
        src_doms: List[str],
        tar_doms: List[str],
        device: str,
        optimizer_name: str,
        nepochs=100,
        lr=0.05,
        lr_patience=5,
        clipgrad=10000,
        momentum=None,
        wd=0,
        wu_nepochs=0,
        wu_lr_factor=1,
        fix_bn=True,
        n_classes=1,
        logger: ExperimentLogger = None,
        metrics: List[str] = [],
        unfreeze_layer: int = 0,
        norm_stats: str = "imagenet",
        batch_size: int = 32,
        not_balanced: bool = False,
        lamb: float = 1.0,
        stage: int = 0,
        disc_lr: float = 2e-4,
        gen_lr: float = 3e-4,
        gan_epochs: int = 200,
        r1_gamma: float = 2,
        gan_batchsize: int = 32,
        gan_gen_lr: float = 2e-4,
        gan_disc_lr: float = 2e-4,
        residual: bool = False,
        attention: bool = False,
        delt: float = 0.1,
        **kwargs
    ):
        Ours.__init__(
            self,
            model,
            patch_dataset,
            src_doms,
            tar_doms,
            device,
            optimizer_name,
            nepochs,
            lr,
            lr_patience,
            clipgrad,
            momentum,
            wd,
            wu_nepochs,
            wu_lr_factor,
            fix_bn,
            n_classes,
            logger,
            metrics,
            unfreeze_layer,
            norm_stats,
            batch_size,
            not_balanced,
            lamb,
            stage,
            disc_lr,
            gen_lr,
            **kwargs
        )
        StyleGAN.__init__(
            self,
            model,
            patch_dataset,
            src_doms,
            tar_doms,
            device,
            n_classes,
            stage,
            gan_disc_lr,
            gan_gen_lr,
            gan_epochs,
            r1_gamma,
            norm_stats,
            gan_batchsize,
            not_balanced,
            residual,
            attention,
            delt,
            **kwargs
        )

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        da_args, extra_args = Ours.extra_parser(args)
        gan_args, extra_args = StyleGAN.extra_parser(extra_args)
        return Namespace(**vars(da_args), **vars(gan_args)), extra_args

    def get_all_loaders(self, name, name_source):
        self.unpack_loaders()
        transform_fun = self.helper_crc if "crctp" in name_source else None
        trn_loader = self.get_loaders(self.train_loaders, name, squeeze=True)
        source_loader = self.make_data_loader(
            self.batch_size,
            get_transform(SET_TYPES[0], self.norm_stats, is_tensor=True),
            transform=transform_fun,
        )
        val_loaders = self.get_loaders(self.val_loaders, name)
        source_val_loaders = {
            name_source: self.make_data_loader(
                self.batch_size,
                get_transform(SET_TYPES[1], self.norm_stats, is_tensor=True),
                transform=transform_fun,
            )
        }
        return trn_loader, source_loader, val_loaders, source_val_loaders

    def train_generative(self, name, distill: bool = False) -> int:
        StyleGAN.train_generative(name, True)

    def load_gan(self, path: str, n_iter: int) -> bool:
        # if an error occurs here: check gan_ckpt_n_iter in the config and then if the checkpoint exists
        if os.path.exists(path):
            gen_fn = list(
                filter(
                    lambda x: str(n_iter) in x and "generator_ema" in x,
                    os.listdir(path),
                )
            )[0]
            disc_fn = list(
                filter(
                    lambda x: str(n_iter) in x and "discriminator" in x,
                    os.listdir(path),
                )
            )[0]
            self.generator_ema_dict = torch.load(os.path.join(path, gen_fn))
            self.discriminator_dict = torch.load(os.path.join(path, disc_fn))
        return os.path.exists(path)

    def transform_model(self):
        head_len_source = self.model.backbone.head_len
        model_name_source = self.model.backbone.name
        self.model_target = OursAddaModel.get_model(
            model_name=model_name_source,
            head_len=head_len_source,
            residual=self.residual,
            attention=self.attention,
            n_classes=self.n_classes,
            stage=self.stage,
        )
        if self.generator_ema_dict:
            self.generator_ema.load_state_dict(deepcopy(self.generator_ema_dict))
            self.model_target.discriminator.set_state_dict(self.discriminator_dict)
        else:
            print("Not using a pretrained generator!")
        if self.source_model_dict:
            self.model.set_state_dict(self.source_model_dict)
        self.model_target.backbone.set_state_dict(self.model.backbone.get_copy())
        self.model_target.classifier.set_state_dict(self.model.classifier.get_copy())
        self.model.freeze_all()

    def update_generative(self, tar_dom):
        pass
