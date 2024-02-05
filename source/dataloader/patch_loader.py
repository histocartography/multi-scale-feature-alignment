from typing import Any, Dict
import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import (
    WeightedRandomSampler,
    SequentialSampler,
)

from source.dataloader.abstract_dataset import AbstractDataset
from source.utils.constants import SET_TYPES
from source.utils.infinite_dataloader import InfiniteDataLoader


def collate_fn(batch):
    images = torch.stack([d[0] for d in batch], axis=0)
    target = torch.stack([d[1] for d in batch], axis=0)
    idx = [d[2] for d in batch]
    return {"images": images, "labels": target, "idx": idx}


class PatchLoaders:
    def __init__(self, batch_size: int, device: str, not_balanced: bool = False):
        self.batch_size: int = batch_size
        self._loaders: Dict[str, Dict[str, DataLoader]] = {}
        self.custom_batch = lambda data: collate_fn(data)
        self.balanced: bool = not not_balanced
        self.samplers: Dict[str, Any] = {}

    def _get_sampler(self, abs_dataset: AbstractDataset, domain: str):
        if self.samplers.get(domain, None) is None:
            if hasattr(abs_dataset, "labels_per_domain"):
                map_cnt = {
                    dom: 1000 * i
                    for i, dom in enumerate(set(abs_dataset.labels_per_domain.keys()))
                }
                labs = [
                    map_cnt[dom] + val
                    for dom, dataset in abs_dataset.labels_per_domain.items()
                    for val in dataset.values()
                ]
            else:
                labs = list(abs_dataset.dataset.values())
            nums, cnts = np.unique(labs, return_counts=True)
            weights = {idx: 1000.0 / cnt for idx, cnt in zip(nums, cnts)}
            weights_t = torch.DoubleTensor([weights[lab] for lab in labs])
            self.samplers[domain] = WeightedRandomSampler(weights_t, len(weights_t))
        return self.samplers[domain]

    def _get_loader(
        self, curr_dataset: AbstractDataset, set_type: str, domain: str
    ) -> InfiniteDataLoader:
        kwargs = {"batch_size": self.batch_size, "num_workers": 2, "pin_memory": False}
        if set_type == SET_TYPES[0] and self.balanced:
            return InfiniteDataLoader(
                curr_dataset,
                collate_fn=self.custom_batch,
                sampler=self._get_sampler(curr_dataset, domain),
                drop_last=True,
                **kwargs
            )
        return InfiniteDataLoader(
            curr_dataset,
            collate_fn=self.custom_batch,
            sampler=SequentialSampler(curr_dataset),
            **kwargs
        )

    def _fill_loaders(self, single_datasets: Dict[str, Dict[str, AbstractDataset]]):
        for set_type, datasets in single_datasets.items():
            self._loaders[set_type] = {
                domain: self._get_loader(dataset, set_type, domain)
                for domain, dataset in datasets.items()
            }

    def get(
        self, single_datasets: Dict[str, Dict[str, AbstractDataset]]
    ) -> Dict[str, Dict[str, DataLoader]]:
        if not self._loaders:
            self._fill_loaders(single_datasets)
        return self._loaders
