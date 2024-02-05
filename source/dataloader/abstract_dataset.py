from __future__ import annotations
import os
from typing import List
from copy import deepcopy

import torch
import pandas as pd
from sklearn.model_selection import train_test_split

from source.utils.constants import MAP_DATASET_STATS, SET_TYPES
import torchvision.transforms as transforms


def get_transform(
    set_type, dataset_stats="imagenet", is_tensor: bool = False, gan_input: bool = False
):
    full_transf = []

    if gan_input:
        full_transf.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop((128, 128)),
            ]
        )
    elif set_type == SET_TYPES[0]:
        full_transf.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomCrop((128, 128)),
            ]
        )
    else:
        full_transf.extend(
            [
                transforms.CenterCrop((128, 128)),
            ]
        )

    if is_tensor:
        full_transf.append(lambda x: x / 255.0)
    else:
        full_transf.append(transforms.ToTensor())

    full_transf.append(transforms.Normalize(**MAP_DATASET_STATS[dataset_stats]))
    transf = transforms.Compose(full_transf)
    return transf


class AbstractDataset(torch.utils.data.Dataset):
    def __init__(self, labels_csv, splits_csv, reduce, base_path):
        if not os.path.exists(labels_csv):
            labels_csv = os.path.join(base_path, labels_csv)
        if not os.path.exists(splits_csv):
            splits_csv = os.path.join(base_path, splits_csv)

        self.labels = pd.read_csv(labels_csv, low_memory=False)
        self.table_splits = pd.read_csv(splits_csv, low_memory=False)
        self.base_path = base_path
        self.reduce = reduce
        self.dataset = {}
        self.set_type = None
        self.transform = None

    def load_dataset(self, slide_ids: List[str]) -> None:
        def tm_sorter(column):
            correspondence = {
                slide_id: order for order, slide_id in enumerate(slide_ids)
            }
            return column.map(correspondence)

        filt_labels = self.labels[self.labels["name_id"].isin(slide_ids)].copy()
        self.dataset = dict(
            zip(
                slide_ids,
                filt_labels.sort_values(by="name_id", key=tm_sorter)["label"].tolist(),
            )
        )
        filt_labels = None

    def shrink_dataset(self):
        if len(self.dataset) > self.reduce:
            X, y = zip(*self.dataset.items())
            _, X_test, _, y_test = train_test_split(
                X, y, test_size=self.reduce / len(self.dataset), stratify=y
            )
            self.dataset = dict(zip(X_test, y_test))

    def _clean_dataset(self, set_type, domain) -> None:
        # filter per type (train, val...) and per name of the dataset (either initial or num)
        table_splits = self.table_splits.copy()
        tmp_dataset = self.table_splits.loc[
            :, table_splits.columns.str.contains(set_type)
        ]
        tmp_dataset = tmp_dataset.loc[:, tmp_dataset.columns.str.contains(domain)]
        assert len(tmp_dataset.columns) == 1, f"Num cols not 1: {tmp_dataset.columns}"
        # del NaNs from csv table
        slide_ids = sorted(
            tmp_dataset.iloc[:, 0].dropna().reset_index(drop=True).tolist()
        )
        self.load_dataset(slide_ids)
        self.set_type = set_type

    def get(self, set_type, domain) -> AbstractDataset:
        self.dataset = {}
        self._clean_dataset(set_type, domain)
        if self.reduce and set_type == SET_TYPES[0]:
            self.shrink_dataset()
        self.keys_dataset = list(self.dataset.keys())
        assert len(self.keys_dataset) > 0, f"{set_type} {domain} not found"
        assert self.set_type is not None, "set_type not set"
        self.domain = domain
        return deepcopy(self)

    def get_classes(self):
        return len(self.labels["label"].unique())

    def init_preprocessing(self, stats_type="imagenet"):
        pass

    def __len__(self) -> int:
        return len(self.dataset)
