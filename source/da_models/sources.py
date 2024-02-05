from typing import List
import time
import torch
from source.da_models.abstract_da import AbstractDAModel
from source.dataloader.patch_dataset import PatchDataset
from source.loggers.exp_logger import ExperimentLogger
from source.networks.resnet_model import ResnetModel


class DAModel(AbstractDAModel):
    """Model implementing DA baseline of training on source only, and testing on target"""

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

    def train_target(self, name: str, name_source: str):
        pass

    def transform_model(self):
        if self.source_model_dict:
            self.model.set_state_dict(self.source_model_dict)
        self.model.freeze_all()

    def criterion(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Returns the loss value"""
        return self.loss_fn(output, target)

    def test_target(self, name, name_source):
        self.model.to(self.device)
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
