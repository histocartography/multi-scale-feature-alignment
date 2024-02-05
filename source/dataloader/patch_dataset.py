from __future__ import annotations
from typing import Tuple
import numpy as np
import torch
import torchvision
import os
from PIL import Image
from source.dataloader.abstract_dataset import AbstractDataset, get_transform
from source.utils.merged_dataset import MergedDataset


class PatchDataset(AbstractDataset):
    def __init__(self, original_csv, split_csv, reduce, base_path):
        super().__init__(original_csv, split_csv, reduce, base_path)
        self.transform = torchvision.transforms.Compose(
            [torchvision.transforms.ToTensor()]
        )

    def init_preprocessing(
        self, stats_type="imagenet", is_tensor: bool = False, gan_input: bool = False
    ):
        self.transform = get_transform(self.set_type, stats_type, is_tensor, gan_input)

    @staticmethod
    def merge(datasets):
        return MergedDataset(datasets)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        filename = self.keys_dataset[idx]
        y = self.dataset[filename]
        if filename is None:
            return self.__getitem__(np.random.randint(0, len(self.dataset)))
        x_np = Image.open(os.path.join(self.base_path, filename))
        return self.transform(x_np), torch.tensor(y), self.keys_dataset[idx]
