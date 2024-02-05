from copy import deepcopy

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
import torch.utils.model_zoo

from source.utils.optimizers import OPTIMIZERS


class FeatureExtractor(torch.nn.Module):
    def __init__(self, net, head_len: int, name: str):
        super(FeatureExtractor, self).__init__()
        self.optimizer: Optimizer = None
        self.lr = 1e-4
        modules = list(net.children())[:-head_len]
        self.model = torch.nn.Sequential(*modules)
        self.pool = torch.nn.AdaptiveAvgPool2d(1)
        self.flat = torch.nn.Flatten()
        self.head_len = head_len
        self.name = name

    def forward(self, x, output_stage=None):
        feat_stage = []
        for name, module in self.model.named_children():
            x = module(x)
            if name >= str(output_stage):
                feat_stage.append(x)

        x = self.pool(x)
        x = self.flat(x)
        return feat_stage, x

    def freeze_bn(self):
        for name, mod in self.named_modules():
            if isinstance(mod, torch.nn.BatchNorm2d):
                for param in mod.parameters():
                    param.requires_grad = False
                mod.eval()

    def freeze(self, unfreeze_layer: int):
        # reset all trainable
        for param in self.parameters():
            param.requires_grad = True

        last_block = []
        # if unfreeze_layer less than whole model and > 0 -> select those to freeze
        # if less than 1 -> freeze all
        if unfreeze_layer < 1:
            last_block = self.model[:]
        elif unfreeze_layer < len(self.model):
            last_block = self.model[:-unfreeze_layer]

        for mod in last_block:
            for param in mod.parameters():
                param.requires_grad = False
            for module in mod.modules():
                module.eval()

    def get_optimizer(self, optimizer_name, original_lr, momentum, wd):
        self.lr = original_lr
        base_params = list(filter(lambda p: p.requires_grad, self.model.parameters()))
        params = [{"params": base_params}]
        kwargs = {"lr": self.lr, "momentum": momentum, "weight_decay": wd}
        kwargs = {key: val for key, val in kwargs.items() if val is not None}
        opts = list(filter(lambda x: x.name == optimizer_name, OPTIMIZERS))
        assert len(opts) == 1, "No optimizer found"
        self.optimizer = opts[0].get(params, **kwargs)

    def get_scheduler(self, n_iters, relative_min_lr):
        self.scheduler = CosineAnnealingLR(
            self.optimizer, T_max=n_iters, eta_min=self.lr * relative_min_lr
        )

    def requires_grad(self, requires_grad: bool = True):
        for param in self.parameters():
            param.requires_grad = requires_grad

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))
