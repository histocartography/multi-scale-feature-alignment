from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR

from source.networks.classifier_head import ClassifierHead
from source.networks.feature_extractor import FeatureExtractor
from source.networks.resnet_model import MODEL_ARCH
from source.utils.constants import DOMS
from source.utils.optimizers import OPTIMIZERS
from source.networks.cond_style_gan import Prologue, Epilogue


class DomainDiscriminator(nn.Module):
    def __init__(self, stage, device, n_cls, residual, attention, doms: int = DOMS):
        super(DomainDiscriminator, self).__init__()
        # prologue depends on replay stage
        assert stage in (0, 5, 6, 7), f"Discriminator not implemented for stage {stage}"
        self.stage = stage

        # interface between prologue and epilogue
        epilogue_in_res = 2
        epilogue_in_nchan = 512

        if stage == 7:
            prologue_in_chan = 512
        elif stage == 6:
            prologue_in_chan = 256
        else:
            prologue_in_chan = 128
        self.prologue = Prologue(
            stage,
            in_n_chan=prologue_in_chan,
            out_nchan=epilogue_in_nchan,
            residual=residual,
            attention=attention,
        )
        # epilogue, with optional mini-batch discrimination layer
        self.epilogue = Epilogue(
            stage,
            epilogue_in_nchan,
            n_cls=n_cls + doms,
            res=epilogue_in_res,
            mbdis_n_chan=1,
        )
        self.to(device)

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def forward(self, x, c):
        x = self.prologue(x)
        x = self.epilogue(x, c)
        return x

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            module.eval()

    def freeze_prologue(self):
        for param in self.prologue.parameters():
            param.requires_grad = False
        for module in self.prologue.modules():
            module.eval()

    def get_optimizer(self, optimizer_name, original_lr, momentum, wd):
        self.lr = original_lr
        base_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        params = [{"params": base_params}]
        kwargs = {
            "lr": self.lr,
            "momentum": momentum,
            "weight_decay": wd,
            "betas": (0.0, 0.99),
        }
        kwargs = {key: val for key, val in kwargs.items() if val is not None}
        opts = list(filter(lambda x: x.name == optimizer_name, OPTIMIZERS))
        assert len(opts) == 1, "No optimizer found"
        self.optimizer = opts[0].get(params, **kwargs)

    def get_scheduler(self, n_iters, relative_min_lr):
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=n_iters, eta_min=self.lr * relative_min_lr
        )

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))


class OursAddaModel(torch.nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        residual: bool,
        attention: bool,
        n_classes: int = 5,
        stage: int = 5,
    ):
        super(OursAddaModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = feature_extractor
        self.output_features = self.backbone(
            torch.ones(1, 3, 224, 224, device=self.device)
        )[-1].shape[-1]
        self.classifier = ClassifierHead(
            output_layers=self.output_features, n_classes=n_classes
        )
        self.stage = stage
        self.discriminator = DomainDiscriminator(
            self.stage,
            self.device,
            n_cls=n_classes,
            attention=attention,
            residual=residual,
        )
        self.n_classes = n_classes
        self.to(self.device)

    def forward(self, x: torch.Tensor, labels=None, feat=None, feat_stage=None, doms=0):
        if feat is None:
            feat_stage, feat = self.backbone(x, self.stage)
        logits = self.classifier(feat)
        labels = (
            torch.argmax(logits.detach().requires_grad_(False), dim=1)
            if labels is None
            else labels
        )
        doms = (
            F.one_hot(
                torch.randint(0, doms, (x.shape[0],), device=self.device), DOMS
            ).float()
            if isinstance(doms, int)
            else doms
        )
        domains = self.discriminator(
            feat_stage,
            torch.cat((F.one_hot(labels, self.n_classes).float(), doms), dim=1),
        )
        return feat, logits, domains.squeeze()

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            module.eval()

    def freeze(self, unfreeze_layer):
        self.backbone.freeze(unfreeze_layer)

    def freeze_bn(self):
        self.backbone.freeze_bn()

    def get_classifier_copy(self):
        return self.classifier.get_copy()

    def get_backbone_copy(self):
        return self.backbone.get_copy()

    def get_optimizer(self, optimizer_name, original_lr, momentum, wd, disc_lr, gen_lr):
        self.backbone.get_optimizer(optimizer_name, gen_lr, momentum, wd)
        self.classifier.get_optimizer(optimizer_name, original_lr, momentum, wd)
        self.discriminator.get_optimizer(optimizer_name, disc_lr, momentum, wd)

    def get_scheduler(self, n_iters, min_lr):
        self.backbone.get_scheduler(n_iters, min_lr)
        self.classifier.get_scheduler(n_iters, min_lr)
        self.discriminator.get_scheduler(n_iters, min_lr)

    def scheduler_step(self):
        self.backbone.scheduler.step()
        self.classifier.scheduler.step()
        self.discriminator.scheduler.step()

    def optimizer_step(self):
        self.backbone.optimizer.step()
        self.classifier.optimizer.step()
        self.discriminator.optimizer.step()

    def optim_zero_grad(self):
        self.backbone.optimizer.zero_grad()
        self.classifier.optimizer.zero_grad()
        self.discriminator.optimizer.zero_grad()

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = self.backbone.to(device)
        self.classifier = self.classifier.to(device)
        self.discriminator = self.discriminator.to(device)

    def requires_grad(self, value):
        for param in self.parameters():
            param.requires_grad = value

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))

    @classmethod
    def get_model(
        cls,
        model_name: str,
        head_len: int,
        residual: bool,
        attention: bool,
        n_classes: int = 5,
        stage: int = 5,
        **kwargs,
    ):
        model = cls(
            deepcopy(
                FeatureExtractor(
                    deepcopy(MODEL_ARCH.get(model_name)),
                    head_len=head_len,
                    name=model_name,
                )
            ),
            n_classes=n_classes,
            stage=stage,
            residual=residual,
            attention=attention,
        )
        return model
