from typing import List
from source.dataloader.abstract_dataset import AbstractDataset
from source.dataloader.patch_dataset import PatchDataset
from source.dataloader.patch_loader import PatchLoaders
from source.utils.constants import SET_TYPES


def get_loaders(
    patch_whole_dataset: AbstractDataset,
    source_domains: List[str],
    target_domains: List[str],
    norm_stats: str = "",
    batch_size: int = 32,
    device: str = "",
    not_balanced: bool = False,
    gan_input: bool = False,
    is_tensor: bool = False,
):
    patch_datasets = {}
    if source_domains is None or target_domains is None:
        print("Warning: no data loaded!")
        print("Disregard warning, if you see this at startup")
        return patch_datasets

    for set_type in SET_TYPES:
        domains = [*source_domains, *target_domains]
        patch_datasets[set_type] = {
            domain: patch_whole_dataset.get(set_type, domain) for domain in domains
        }
        for dataset in patch_datasets[set_type].values():
            dataset.init_preprocessing(
                stats_type=norm_stats, gan_input=gan_input, is_tensor=is_tensor
            )

        # merge source datasets for train only
        if set_type == SET_TYPES[0] and len(source_domains) > 0:
            src_dom = "_".join(source_domains)
            list_dataset = [patch_datasets[set_type].pop(dom) for dom in source_domains]
            patch_datasets[set_type][src_dom] = PatchDataset.merge(list_dataset)

    patch_loaders = PatchLoaders(
        batch_size=batch_size, device=device, not_balanced=not_balanced
    ).get(patch_datasets)
    return patch_loaders
