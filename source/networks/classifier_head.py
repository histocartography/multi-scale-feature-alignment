from copy import deepcopy

import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.optimizer import Optimizer
import torch.utils.model_zoo

from source.utils.optimizers import OPTIMIZERS


class ClassifierHead(torch.nn.Module):
    def __init__(self, output_layers: int, n_classes: int = 5):
        super(ClassifierHead, self).__init__()
        self.n_classes = n_classes
        self.optimizer: Optimizer = None
        linear1 = torch.nn.Linear(output_layers, output_layers)
        do = torch.nn.Dropout(inplace=True)
        relu = torch.nn.ReLU(inplace=True)
        linear2 = torch.nn.Linear(output_layers, n_classes)
        self.model = torch.nn.Sequential(*[linear1, do, relu, linear2])

    def forward(self, x):
        return self.model(x)

    def freeze(self, unfreeze_layer: int = 0):
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            module.eval()

    def relocate(self):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(device)

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

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))

    def requires_grad(self, requires_grad: bool = True):
        for param in self.parameters():
            param.requires_grad = requires_grad
