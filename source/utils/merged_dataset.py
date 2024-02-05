from typing import Iterable
from torch.utils.data.dataset import ConcatDataset

from source.dataloader.abstract_dataset import AbstractDataset, get_transform


class MergedDataset(ConcatDataset):
    def __init__(self, datasets: Iterable[AbstractDataset]) -> None:
        super().__init__(datasets)
        self.labels = {k: v for dataset in datasets for k, v in dataset.dataset.items()}
        self.labels_per_domain = {
            dataset.domain: {k: v for k, v in dataset.dataset.items()}
            for dataset in datasets
        }
        self.transform = None

    def init_preprocessing(
        self, stats_type="imagenet", is_tensor: bool = False, gan_input: bool = False
    ):
        for dataset in self.datasets:
            dataset.transform = get_transform(
                dataset.set_type, stats_type, is_tensor, gan_input
            )
