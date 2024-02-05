from copy import deepcopy

import torch
import torch.utils.model_zoo
import torchvision
from torchvision.models import ResNet18_Weights, ResNet50_Weights

from source.networks.classifier_head import ClassifierHead
from source.networks.feature_extractor import FeatureExtractor


#
class ResnetModel(torch.nn.Module):
    def __init__(
        self,
        feature_extractor: FeatureExtractor,
        classifier: bool = False,
        n_classes: int = 5,
    ):
        super(ResnetModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.backbone = feature_extractor
        self.backbone = self.backbone.to(self.device)
        self.classifier = None
        self.output_features = self.backbone(
            torch.ones((1, 3, 224, 224), device=self.device)
        )[-1].shape[-1]
        if classifier:
            self.classifier = ClassifierHead(
                output_layers=self.output_features, n_classes=n_classes
            )
            self.classifier.to(self.device)

    def forward(self, x, output_stage=None):
        feat_stage, feat = self.backbone(x, output_stage)
        logits = None
        if self.classifier:
            logits = self.classifier(feat)
        return feat, logits, feat_stage

    def requires_grad(self, value: bool):
        for part in [self.backbone, self.classifier]:
            if part is None:
                continue
            for param in part.parameters():
                param.requires_grad = value

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False
        for module in self.modules():
            module.eval()

    def freeze(self, unfreeze_layer):
        self.backbone.freeze(unfreeze_layer)

    def get_classifier_copy(self):
        return self.classifier.get_copy()

    def get_backbone_copy(self):
        return self.backbone.get_copy()

    def get_optimizer(self, optimizer_name, original_lr, momentum, wd):
        self.backbone.get_optimizer(optimizer_name, original_lr, momentum, wd)
        if self.classifier:
            self.classifier.get_optimizer(optimizer_name, original_lr, momentum, wd)

    def get_scheduler(self, n_iters, min_lr):
        self.backbone.get_scheduler(n_iters, min_lr)
        if self.classifier:
            self.classifier.get_scheduler(n_iters, min_lr)

    def scheduler_step(self):
        self.backbone.scheduler.step()
        if self.classifier:
            self.classifier.scheduler.step()

    def optimizer_step(self):
        self.backbone.optimizer.step()
        if self.classifier:
            self.classifier.optimizer.step()

    def optim_zero_grad(self):
        self.backbone.optimizer.zero_grad()
        if self.classifier:
            self.classifier.optimizer.zero_grad()

    def relocate(self):
        self.backbone = self.backbone.to(self.device)
        if self.classifier:
            self.classifier = self.classifier.to(self.device)

    def get_copy(self):
        """Get weights from the model"""
        return deepcopy(self.state_dict())

    def set_state_dict(self, state_dict):
        """Load weights into the model"""
        self.load_state_dict(deepcopy(state_dict))

    @classmethod
    def get_model(
        cls, model_name: str, head_len: int, n_classes: int = 5, classifier: bool = True
    ):
        model = cls(
            FeatureExtractor(
                MODEL_ARCH[model_name], name=model_name, head_len=head_len
            ),
            classifier=classifier,
            n_classes=n_classes,
        )
        return model


MODEL_ARCH = {
    "resnet18": torchvision.models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1),
    "resnet50": torchvision.models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1),
}
