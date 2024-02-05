import os
import time
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
import torch.nn as nn
from argparse import ArgumentParser
import torch.nn.functional as F
from sklearn.manifold import TSNE
from source.dataloader.get_loaders import get_loaders
from source.dataloader.patch_dataset import PatchDataset

from source.loggers.exp_logger import ExperimentLogger
from source.utils.constants import ITER_EPOCHS, MAX_ITERS, SET_TYPES
from source.utils.infinite_dataloader import InfiniteDataLoader
from source.utils.metrics import Metrics
from source.networks.resnet_model import ResnetModel


class AbstractDAModel(object):
    """Basic class for implementing incremental learning approaches"""

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
    ):

        self.model = model

        self.device = device
        self.nepochs = nepochs
        self.lr = lr
        self.lr_patience = lr_patience
        self.clipgrad = clipgrad
        self.momentum = momentum
        self.wd = wd
        self.logger: ExperimentLogger = logger
        self.metrics = Metrics(metrics)
        self.warmup_epochs = wu_nepochs
        self.warmup_lr_factor = wu_lr_factor
        self.warmup_loss = nn.CrossEntropyLoss()
        self.fix_bn = fix_bn
        self.n_classes = n_classes
        self.optimizer_name = optimizer_name
        self.scheduler = None
        self.loss_fn = nn.CrossEntropyLoss()
        self.unfreeze_layer = unfreeze_layer

        self.patch_dataset = patch_dataset
        self.src_doms = list(src_doms) if src_doms is not None else None
        self.tar_doms = list(tar_doms) if src_doms is not None else None
        self.batch_size = batch_size
        self.norm_stats = norm_stats
        self.not_balanced = not_balanced

        self.loaders_kwargs = dict(
            norm_stats=norm_stats,
            batch_size=batch_size,
            device=device,
            not_balanced=not_balanced,
        )
        self.patch_loaders: Dict[str, Dict[str, InfiniteDataLoader]] = get_loaders(
            self.patch_dataset, self.src_doms, self.tar_doms, **self.loaders_kwargs
        )
        self.unpack_loaders()
        self.step = 0
        self.source_model_dict = None
        if model is not None:
            self._relocate()
        self.old_tar_dom = []
        self.generator_ema_dict = {}
        self.discriminator_dict = {}

    @staticmethod
    def extra_parser(args):
        """Returns a parser containing the approach specific parameters"""
        parser = ArgumentParser()
        return parser.parse_known_args(args)

    def _relocate(self):
        self.model.relocate()

    def unpack_loaders(self):
        self.test_loaders = self.patch_loaders.get(SET_TYPES[2], None)
        self.val_loaders = self.patch_loaders.get(SET_TYPES[1], None)
        self.train_loaders = self.patch_loaders.get(SET_TYPES[0], None)

    @staticmethod
    def get_loaders(
        loaders: Dict[str, InfiniteDataLoader], name: str, squeeze: bool = False
    ) -> Union[Dict[str, InfiniteDataLoader], InfiniteDataLoader]:
        loaders_filt = dict(filter(lambda x: x[0] in name, loaders.items()))
        if len(loaders_filt) == 1 and squeeze:
            return loaders_filt[name]
        return loaders_filt

    def get_all_loaders(self, name: str, name_source: str):
        self.unpack_loaders()
        trn_loader = self.get_loaders(self.train_loaders, name, squeeze=True)
        source_loader = self.get_loaders(self.train_loaders, name_source, squeeze=True)
        val_loaders = self.get_loaders(self.val_loaders, name)
        source_val_loaders = self.get_loaders(self.val_loaders, name_source)
        return trn_loader, source_loader, val_loaders, source_val_loaders

    def train_source(self, name: str, warmup: bool = False):
        """Contains the epochs loop"""
        # housekeeping for training init
        self._relocate()
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()
        # adapt epochs, lr, eta if warmup
        epochs = self.nepochs if not warmup else self.warmup_epochs
        lr = self.lr if not warmup else self.warmup_lr_factor * self.lr
        eta_scheduler = 0.05 if not warmup else 1 / self.warmup_lr_factor

        # for compatibility use it with dummy "source" -> source is already 'name'
        trn_loader, _, val_loaders, _ = self.get_all_loaders(name, "")
        self.model.freeze(self.unfreeze_layer)
        self.model.get_optimizer(self.optimizer_name, lr, self.momentum, self.wd)
        self.model.get_scheduler(ITER_EPOCHS * epochs, eta_scheduler)

        # loop over epochs
        for e in range(epochs):
            # Training
            clock0 = time.time()
            self.model.train()
            self.model.freeze(self.unfreeze_layer)
            self.model.zero_grad()
            total_loss_ce = []
            for i in range(ITER_EPOCHS):
                data, _ = trn_loader.fetch()
                logits = self.model(data["images"].to(self.device))[1]
                loss_ce = self.criterion(logits, data["labels"].to(self.device))
                loss_ce.backward()

                self.model.optimizer_step()
                self.model.scheduler_step()
                self.model.optim_zero_grad()

                total_loss_ce.append(loss_ce.cpu().detach().item())

                self.step += 1

            # log and print training
            dict_train_loss = {"loss_total": np.mean(total_loss_ce)}
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

            # Validation + log and printing validation
            loss_list = []
            for name_val, val_loader in val_loaders.items():
                clock3 = time.time()
                dict_valid_loss, valid_metrics = self.eval_source(name_val, val_loader)
                clock4 = time.time()
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
                    [
                        f"{name}={loss:.3f}"
                        for name, loss in dict_valid_loss.items()
                        if loss
                    ]
                )
                valid_scores_str = " ".join(
                    [f"{name}={score:.4f}" for name, score in valid_metrics.items()]
                )
                print(
                    f" Valid {name_val}: time={clock4 - clock3:5.1f}s losses: {valid_loss_str}, scores: {valid_scores_str} |",
                    end=" ",
                )
                loss_list.append(dict_valid_loss["loss_total"])
            loss_total = np.average(
                loss_list,
                weights=[len(val_loader) for val_loader in val_loaders.values()],
            )

            # Adapt learning rate - patience scheme - early stopping regularization
            if loss_total < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = loss_total
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(" *", end="")
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    self.model.set_state_dict(best_model)
                    print()
                    break
            print()
        self.model.set_state_dict(best_model)

    def eval_source(
        self, name: str, val_loader=None, testing: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Contains the evaluation code"""
        # import psutil
        # process = psutil.Process(os.getpid())
        if val_loader is None:
            if testing:
                val_loader = self.get_loaders(self.test_loaders, name, squeeze=True)
            else:
                val_loader = self.get_loaders(self.val_loaders, name, squeeze=True)

        with torch.no_grad():
            self.model.eval()
            val_probs, val_targets, total_loss_ce = [], [], []
            val_loader.init()
            for _ in range(MAX_ITERS):
                data, isok = val_loader.fetch()
                if not isok:
                    break
                logits = self.model(data["images"].to(self.device))[1]
                loss_ce = self.criterion(logits, data["labels"].to(self.device))
                val_probs.append(F.softmax(logits, dim=1).detach().cpu().numpy())
                val_targets.append(data["labels"].detach().cpu().numpy())
                total_loss_ce.append(loss_ce.detach().cpu().item())
            metrics = self.calculate_metrics(
                np.concatenate(val_probs, axis=0), np.concatenate(val_targets, axis=0)
            )

        return {"loss_total": np.mean(total_loss_ce)}, metrics

    def test_source(self, name: str):
        self._relocate()
        for name_test, test_loader in self.get_loaders(self.test_loaders, name).items():
            dict_valid_loss, valid_metrics = self.eval_source(name_test, test_loader)
            for loss_name, loss_val in dict_valid_loss.items():
                self.logger.log_scalar(
                    task=name_test,
                    iter=self.step,
                    name=loss_name,
                    value=loss_val,
                    group="test",
                )
            for name_metric, score_metric in valid_metrics.items():
                self.logger.log_scalar(
                    task=name_test,
                    iter=self.step,
                    name=name_metric,
                    value=score_metric,
                    group="test",
                )

    def train_target(self, name: str, name_source: str):
        """Contains the epochs loop"""
        self._relocate()
        best_loss = np.inf
        patience = self.lr_patience
        best_model = self.model.get_copy()

        trn_loader = self.get_loaders(self.train_loaders, name, squeeze=True)

        self.model.freeze(self.unfreeze_layer)
        self.model.get_optimizer(self.optimizer_name, self.lr, self.momentum, self.wd)
        self.model.get_scheduler(ITER_EPOCHS * self.nepochs, 0.05)
        # loop over epochs
        for e in range(self.nepochs):
            # Train
            clock0 = time.time()
            self.model.train()
            self.model.freeze(self.unfreeze_layer)
            self.model.zero_grad()
            total_loss_ce = []
            for i in range(ITER_EPOCHS):
                data, _ = trn_loader.fetch()
                logits = self.model(data["images"].to(self.device))[-1]
                loss_ce = self.criterion(logits, data["labels"].to(self.device))
                loss_ce.backward()

                self.model.optimizer_step()
                self.model.scheduler_step()
                self.model.optim_zero_grad()

                total_loss_ce.append(loss_ce.cpu().detach().item())
                self.step += 1

            dict_train_loss = {"loss_total": np.mean(total_loss_ce)}
            clock1 = time.time()
            print(
                f"|| Epoch {e:3d} || time={clock1 - clock0:5.1f}s | Train: skip eval |",
                end="",
            )
            clock3 = time.time()
            dict_valid_loss, valid_metrics = self.eval_target(name, name_source)
            clock4 = time.time()
            for loss_name, loss_val in dict_train_loss.items():
                self.logger.log_scalar(
                    task=name,
                    iter=self.step,
                    name=loss_name,
                    value=loss_val,
                    group="train",
                )
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

            # Adapt learning rate - patience scheme - early stopping regularization
            if dict_valid_loss["loss_total"] < best_loss:
                # if the loss goes down, keep it as the best model and end line with a star ( * )
                best_loss = dict_valid_loss["loss_total"]
                best_model = self.model.get_copy()
                patience = self.lr_patience
                print(" *", end="")
            else:
                # if the loss does not go down, decrease patience
                patience -= 1
                if patience <= 0:
                    self.model.set_state_dict(best_model)
                    print()
                    break
            print()
        self.model.set_state_dict(best_model)

    def eval_target(
        self, name: str, name_source, testing: bool = False
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        return self.eval_source(name, testing=testing)

    def test_target(self, name, name_source):
        self.model.to(self.device)
        self.model_target.to(self.device)
        print(f"test on target {name}")
        # same test as source testing, just different test_loader
        clock3 = time.time()
        dict_valid_loss, valid_metrics = self.eval_target(name, name_source, True)
        clock4 = time.time()
        print(f"elapsed {clock4 - clock3:5.1f}s")
        for loss_name, loss_val in dict_valid_loss.items():
            self.logger.log_scalar(
                task=name, iter=self.step, name=loss_name, value=loss_val, group="test"
            )
        for name_metric, score_metric in valid_metrics.items():
            self.logger.log_scalar(
                task=name,
                iter=self.step,
                name=name_metric,
                value=score_metric,
                group="test",
            )

    def test_final(self, name_source):
        print("test on source")
        self.test_source(name_source)
        for k in self.test_loaders.keys():
            if k not in name_source:
                self.test_target(k, name_source)

    def save_model(self, path, sequence):
        torch.save(
            self.model.get_copy(), os.path.join(path, f"{'_'.join(sequence)}.pt")
        )

    def calculate_metrics(
        self, probs: np.ndarray, targets: np.ndarray
    ) -> Dict[str, float]:
        np_probs = probs
        np_targets = targets
        return self.metrics.get_scores(np_targets, np_probs)

    def calculate_im(
        self, probs_src: np.ndarray, probs_tar: np.ndarray, name: str
    ) -> Tuple[float, float]:
        # H(mean(p_i)) - mean(H(p_i))
        mean_probs_src = np.mean(probs_src, axis=0)
        mean_probs_tar = np.mean(probs_tar, axis=0)
        im_src = np.sum(
            -mean_probs_src * np.log(mean_probs_src + 1e-8), axis=0
        ) - np.mean(np.sum(-probs_src * np.log(probs_src + 1e-8), axis=1), axis=0)
        im_tar = np.sum(
            -mean_probs_tar * np.log(mean_probs_tar + 1e-8), axis=0
        ) - np.mean(np.sum(-probs_tar * np.log(probs_tar + 1e-8), axis=1), axis=0)
        return im_src, im_tar

    def calculate_distance(self, src_feats, tar_feats, src_labels, tar_labels):
        dist_dict = {}
        for label in np.intersect1d(np.unique(src_labels), np.unique(tar_labels)):
            dist_dict[f"wass_dist cl{label}"] = (
                self.sinkhorn(
                    torch.tensor(src_feats[src_labels == label], device=self.device),
                    torch.tensor(tar_feats[tar_labels == label], device=self.device),
                )[0]
                .detach()
                .cpu()
                .item()
            )
        return dist_dict

    def criterion(
        self,
        output: torch.Tensor,
        target: torch.Tensor,
        output_old: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Returns the loss value"""
        raise NotImplementedError

    def transform_model(self):
        pass

    def load_gan(self, gan_path: str, n_iter: int = 250) -> bool:
        return True

    def load_model(self, path):
        if os.path.exists(path):
            self.source_model_dict = torch.load(path)
        return os.path.exists(path)

    def update_generative(self, name, distill=False):
        pass

    def adapt_sources(self, name: str):
        self.old_tar_dom.append(name)
