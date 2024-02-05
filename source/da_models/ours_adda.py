from collections import OrderedDict, defaultdict
import os
from typing import Dict, List, Tuple
import time
import numpy as np
import torch
import torch.nn.functional as F
from source.da_models.abstract_da import AbstractDAModel
from source.dataloader.patch_dataset import PatchDataset
from source.loggers.exp_logger import ExperimentLogger
from source.networks.ours_adda_model import DOMS, OursAddaModel
from source.networks.resnet_model import ResnetModel
from argparse import ArgumentParser

from source.networks.resnet_model import ResnetModel
from source.utils.constants import (
    ITER_EPOCHS,
    ITER_OURS,
    MAX_ITERS,
    MIN_EPOCHS,
)


class DAModel(AbstractDAModel):
    """Model implementing our proposed multi-scale feature alignment for domain adaptation."""

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
        stage: int = 5,
        disc_lr: float = 1e-4,
        gen_lr: float = 1e-4,
        **kwargs,
    ):
        super(DAModel, self).__init__(
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
        )

        self.lamb: float = lamb
        self.stage: int = stage
        self.disc_lr: float = disc_lr
        self.gen_lr: float = gen_lr
        self.bce = torch.nn.BCEWithLogitsLoss()

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        parser.add_argument(
            "--lamb",
            default=1.0,
            type=float,
            required=False,
            help="tradeoff domain - classifier",
        )
        parser.add_argument(
            "--disc-lr",
            default=1e-4,
            type=float,
            required=False,
            help="tradeoff domain - classifier",
        )
        parser.add_argument(
            "--gen-lr",
            default=1e-4,
            type=float,
            required=False,
            help="tradeoff domain - classifier",
        )
        return parser.parse_known_args(args)

    def _relocate_adda(self):
        self.model_target.relocate()

    def train_target(self, name: str, name_source: str):
        """Contains the epochs loop"""
        self._relocate()
        self._relocate_adda()
        self.model.freeze_all()
        self.model.eval()
        self.true_doms = int(len(self.old_tar_dom) + len(self.src_doms))
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model_target.get_copy()
        trn_loader, source_loader, self.val_loaders, self.source_val_loaders = (
            self.get_all_loaders(name, name_source + "_" + "_".join(self.old_tar_dom))
        )
        self.model_target.freeze(self.unfreeze_layer)
        self.model_target.get_optimizer(
            self.optimizer_name,
            self.lr,
            self.momentum,
            self.wd,
            self.disc_lr,
            self.gen_lr,
        )
        self.model_target.get_scheduler(ITER_EPOCHS * self.nepochs, 0.05)
        # loop over epochs
        max_f1 = 0.0
        for e in range(self.nepochs):
            # Training
            clock0 = time.time()
            self.model_target.train()
            self.model_target.freeze(self.unfreeze_layer)
            self.model_target.optim_zero_grad()
            total_loss_ce, total_loss_disc_t, total_loss_disc_s, total_loss_gen = (
                [],
                [],
                [],
                [],
            )
            total_loss_r1 = []

            loss_gen, loss_ce = torch.tensor(0.0), torch.tensor(0.0)
            for _ in range(ITER_OURS):
                # training model using source data
                with torch.no_grad():
                    data_source, _ = source_loader.fetch()
                    data_target, _ = trn_loader.fetch()
                    source_domain_label = torch.ones(
                        data_source["images"].shape[0], device=self.device
                    )
                    target_domain_label = torch.zeros(
                        data_target["images"].shape[0], device=self.device
                    )
                    src_stage, src_feat = self.model.backbone(
                        data_source["images"].to(self.device), self.stage
                    )
                    src_logits = self.model.classifier(src_feat)
                    target_doms = F.one_hot(
                        torch.randint(
                            0,
                            self.true_doms,
                            (data_target["labels"].shape[0],),
                            device=self.device,
                        ),
                        DOMS,
                    ).float()

                self.model_target.backbone.requires_grad(True)
                self.model_target.classifier.requires_grad(False)
                self.model_target.discriminator.requires_grad(False)
                _, _, target_domain_output = self.model_target(
                    data_target["images"].to(self.device), doms=target_doms
                )
                loss_gen = self.criterion(target_domain_output, 1 - target_domain_label)
                loss_gen.backward()

                self.model_target.backbone.requires_grad(True)
                self.model_target.classifier.requires_grad(True)
                self.model_target.discriminator.requires_grad(False)
                _, classifier_output, _ = self.model_target(
                    data_source["images"].to(self.device),
                    doms=data_source["domains"].to(self.device),
                )
                loss_ce = F.l1_loss(classifier_output, src_logits)
                (self.lamb * loss_ce).backward()
                self.model_target.optimizer_step()
                self.model_target.optim_zero_grad()

                # train domain discriminator, source_features from (frozen / no_grad) old model
                self.model_target.backbone.requires_grad(False)
                self.model_target.classifier.requires_grad(False)
                self.model_target.discriminator.requires_grad(True)

                _, tar_logits, target_domain_output = self.model_target(
                    data_target["images"].to(self.device), doms=target_doms
                )
                loss_disc_t = self.criterion(target_domain_output, target_domain_label)
                loss_disc_t.backward()

                tar_stage, _ = self.model_target.backbone(
                    data_target["images"].to(self.device), self.stage
                )
                loss_disc_r1 = self._get_r1_loss(
                    tar_stage,
                    torch.cat(
                        (
                            self._to_one_hot(torch.argmax(tar_logits.detach(), dim=1)),
                            target_doms,
                        ),
                        dim=1,
                    ),
                    self.step,
                    False,
                )
                if isinstance(loss_disc_r1, torch.Tensor):
                    loss_disc_r1.backward()
                    total_loss_r1.append(loss_disc_r1.detach().cpu().item())
                _, _, source_domain_output = self.model_target(
                    data_source["images"].to(self.device),
                    labels=data_source["labels"].to(self.device),
                    feat=src_feat,
                    feat_stage=src_stage,
                    doms=data_source["domains"].to(self.device),
                )
                loss_disc_s = self.criterion(source_domain_output, source_domain_label)
                loss_disc_s.backward()

                self.model_target.optimizer_step()
                self.model_target.optim_zero_grad()

                total_loss_disc_t.append(loss_disc_t.detach().cpu().item())
                total_loss_disc_s.append(loss_disc_s.detach().cpu().item())

                total_loss_gen.append(loss_gen.detach().cpu().item())
                total_loss_ce.append(loss_ce.detach().cpu().item())
                self.step += 1

            dict_train_loss = {
                "loss_total": self.lamb * np.mean(total_loss_ce)
                + (
                    (np.mean(total_loss_disc_t) + np.mean(total_loss_disc_s)) / 2
                    + np.mean(total_loss_gen)
                )
                / 2,
                "loss_ce": np.mean(total_loss_ce),
                "loss_disc_t": np.mean(total_loss_disc_t),
                "loss_disc_s": np.mean(total_loss_disc_s),
                "loss_gen": np.mean(total_loss_gen),
                "r1_loss": np.mean(total_loss_r1),
            }

            # Log and print training
            clock1 = time.time()
            print(
                f"|| Epoch {e:3d} || time={clock1 - clock0:5.1f}s | Train: skip eval |",
                end="",
            )
            for loss_name, loss_val in dict_train_loss.items():
                self.logger.log_scalar(
                    task=name,
                    iter=self.step,
                    name=loss_name,
                    value=loss_val,
                    group="train",
                )

            # Valid
            clock3 = time.time()
            dict_valid_loss, valid_metrics = self.eval_target(name, name_source)
            clock4 = time.time()

            # Log and print validation
            for loss_name, loss_val in dict_valid_loss.items():
                self.logger.log_scalar(
                    task=name,
                    iter=self.step,
                    name=loss_name,
                    value=loss_val,
                    group="valid",
                )
            for name_metric, score_metric in valid_metrics.items():
                self.logger.log_scalar(
                    task=name,
                    iter=self.step,
                    name=name_metric,
                    value=score_metric,
                    group="valid",
                )
            valid_loss_str = " ".join(
                [f"{name}={loss:.3f}" for name, loss in dict_valid_loss.items() if loss]
            )
            valid_scores_str = " ".join(
                [f"{name}={score:.4f}" for name, score in valid_metrics.items()]
            )
            print(
                f" Valid {name}: time={clock4 - clock3:5.1f}s losses: {valid_loss_str}, scores: {valid_scores_str} |",
                end="",
            )

            if valid_metrics["f1"] > max_f1:
                max_f1 = valid_metrics["f1"]
            self.logger.log_scalar(
                task=name, iter=self.step, name="f1tar_max", value=max_f1, group="valid"
            )

            # Adapt learning rate - patience scheme - early stopping regularization
            if e > MIN_EPOCHS:
                if dict_valid_loss["im"] < best_loss:
                    # if the loss goes down, keep it as the best model and end line with a star ( * )
                    best_loss = dict_valid_loss["im"]
                    best_model = self.model_target.get_copy()
                    patience = self.lr_patience
                    print(" *", end="")
                else:
                    # if the loss does not go down, decrease patience
                    patience -= 1
                    if patience <= 0:
                        print()
                        break
            print()

        self.model_target.set_state_dict(best_model)
        original_model_dict = OrderedDict(
            filter(lambda x: "discriminator" not in x[0], best_model.items())
        )
        self.model.set_state_dict(original_model_dict)

    def eval_target(
        self, name, name_source, testing: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        # get val/test on tar and src, if testing mode val == test (manage with ifs in code)
        test_loader = self.get_loaders(self.test_loaders, name, squeeze=True)

        val_loader = self.val_loaders[name] if not testing else test_loader
        source_test_loaders = self.get_loaders(
            self.test_loaders, "_".join([name_source, *self.old_tar_dom])
        )
        source_val_loaders = (
            self.source_val_loaders if not testing else source_test_loaders
        )

        f1_src = {}
        val_loader.init()
        test_loader.init()
        fast = True
        with torch.no_grad():
            self.model_target.eval()
            self.model.eval()
            tar_probs, tar_targets = [], []
            (
                total_loss_ce,
                total_loss_gen,
                total_loss_disc_t,
                total_loss_disc_s,
                src_total_loss_dist,
            ) = ([], [], [], [], [])
            true_src_probs, src_probs, true_src_targets = (
                defaultdict(list),
                [],
                defaultdict(list),
            )

            for _ in range(MAX_ITERS):
                data, isok = val_loader.fetch()
                if not isok:
                    break

                # calc loss_disc given current target input and some sources
                for i, (key, source_val_loader) in enumerate(
                    source_val_loaders.items()
                ):
                    src_data, _ = source_val_loader.fetch()
                    src_stage, src_feat = self.model.backbone(
                        src_data["images"].to(self.device), self.stage
                    )

                    # calc discriminator loss
                    doms = src_data.get(
                        "domains",
                        F.one_hot(
                            torch.tensor(
                                i, device=self.device, dtype=torch.int64
                            ).repeat(src_data["images"].shape[0]),
                            DOMS,
                        ).float(),
                    ).to(self.device)
                    _, logits, source_domain_output = self.model_target(
                        src_data["images"].to(self.device),
                        labels=src_data["labels"].to(self.device),
                        feat=src_feat,
                        feat_stage=src_stage,
                        doms=doms,
                    )
                    loss_disc_s = self.criterion(
                        source_domain_output,
                        torch.ones(src_data["images"].shape[0]).to(self.device),
                    )
                    total_loss_disc_s.append(loss_disc_s.detach().cpu().item())
                    src_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())

                    # calculate distillation loss source-target
                    src_logits = self.model.classifier(src_feat)
                    _, classifier_output, _ = self.model_target(
                        src_data["images"].to(self.device), doms=doms
                    )
                    loss_dist = F.l1_loss(classifier_output, src_logits)
                    src_total_loss_dist.append(loss_dist.detach().cpu().item())

                    # register f1 on original source
                    if not fast:
                        src_k = (
                            np.random.choice(list(source_test_loaders.keys()))
                            if not testing
                            else key
                        )
                        true_src_data = (
                            source_test_loaders[src_k].fetch()[0]
                            if not testing
                            else src_data
                        )
                        true_src_probs[src_k].append(
                            F.softmax(
                                self.model_target(
                                    true_src_data["images"].to(self.device)
                                )[1],
                                dim=1,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                        )
                        true_src_targets[src_k].append(
                            true_src_data["labels"].detach().cpu().numpy()
                        )

                # calc fooling ability of target feature extractor
                _, logits, domain = self.model_target(
                    data["images"].to(self.device), doms=self.true_doms
                )
                loss_disc_t = self.criterion(
                    domain, torch.zeros(data["images"].shape[0], device=self.device)
                )
                loss_gen = self.criterion(
                    domain, torch.ones(data["images"].shape[0], device=self.device)
                )
                total_loss_disc_t.append(loss_disc_t.detach().cpu().item())
                total_loss_gen.append(loss_gen.detach().cpu().item())

                # calc classification on testing data
                test_data = test_loader.fetch()[0] if not testing and not fast else data
                _, logits, domain = self.model_target(
                    test_data["images"].to(self.device), doms=self.true_doms
                )
                loss_ce = self.criterion(logits, test_data["labels"].to(self.device))
                total_loss_ce.append(loss_ce.detach().cpu().item())
                tar_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
                tar_targets.append(test_data["labels"].detach().cpu().numpy())

            metrics = self.calculate_metrics(
                np.concatenate(tar_probs, axis=0), np.concatenate(tar_targets, axis=0)
            )
            if not fast:
                f1_src = {
                    f"f1src_{src_k}": self.calculate_metrics(
                        np.concatenate(true_src_probs[src_k], axis=0),
                        np.concatenate(true_src_targets[src_k], axis=0),
                    )["f1"]
                    for src_k in true_src_probs.keys()
                }
            im_src, im_tar = self.calculate_im(
                np.concatenate(src_probs, axis=0),
                np.concatenate(tar_probs, axis=0),
                name,
            )

        return {
            "loss_total": -(np.mean(total_loss_disc_s) + np.mean(total_loss_disc_t)) / 2
            + np.mean(total_loss_gen)
            + self.lamb * np.mean(src_total_loss_dist),
            "loss_ce": np.mean(total_loss_ce),
            "loss_disc_t": np.mean(total_loss_disc_t),
            "loss_disc_s": np.mean(total_loss_disc_s),
            "loss_dist": np.mean(src_total_loss_dist),
            "loss_gen": np.mean(total_loss_gen),
            "im_src": im_src,
            "im_tar": im_tar,
            "im": -im_tar + self.lamb * np.mean(src_total_loss_dist),
            **f1_src,
        }, metrics

    def load_gan(self, path: str, n_iter: int = 250) -> bool:
        if os.path.exists(path):
            disc_fn = list(
                filter(
                    lambda x: str(n_iter) in x and "discriminator" in x,
                    os.listdir(path),
                )
            )[0]
            self.discriminator_dict = torch.load(os.path.join(path, disc_fn))
        return os.path.exists(path)

    def transform_model(self):
        head_len_source = self.model.backbone.head_len
        model_name_source = self.model.backbone.name
        self.model_target = OursAddaModel.get_model(
            model_name_source, head_len_source, self.n_classes, self.stage
        )
        if self.source_model_dict:
            self.model.set_state_dict(self.source_model_dict)
        self.model_target.backbone.set_state_dict(self.model.backbone.get_copy())
        self.model_target.classifier.set_state_dict(self.model.classifier.get_copy())
        self.model.freeze_all()
        self.model.eval()

    def criterion(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns the loss value"""
        if output.ndim == 2:
            return self.loss_fn(output, target.long())
        elif output.ndim == 1:
            return self.bce(output, target.float())

    def adapt_sources(self, name: str):
        self.old_tar_dom.append(name)

    @staticmethod
    # if crctp and class adipose or backgroung => shift to the previous one
    # we assume that in k19 crc k16 sequence: crc is at the lat position in the
    # conditioning code. In k19 k16 crc, crc never generated
    def helper_crc(xs, pos_crc=8):
        for x in xs:
            if x[pos_crc] and (x[6] or x[5]):
                x[-2], x[pos_crc] = x[-1], x[-2]
        return xs
