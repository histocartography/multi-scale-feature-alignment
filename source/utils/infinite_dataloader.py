from typing import Any, Callable, List, Optional, Sequence, TypeVar
from torch.utils.data.dataloader import DataLoader, T_co
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import Sampler

# just to simulate torch original DataLoader
T_co = TypeVar("T_co", covariant=True)
T = TypeVar("T")
_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[List[T]], Any]


class InfiniteDataLoader(DataLoader):
    def __init__(
        self,
        dataset: Dataset[T_co],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        batch_sampler: Optional[Sampler[Sequence]] = None,
        num_workers: int = 0,
        collate_fn: Optional[_collate_fn_t] = None,
        pin_memory: bool = False,
        drop_last: bool = False,
        timeout: float = 0,
        worker_init_fn: Optional[_worker_init_fn_t] = None,
        multiprocessing_context=None,
        generator=None,
        *,
        prefetch_factor: int = 2,
        persistent_workers: bool = False
    ):
        super().__init__(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler=batch_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            prefetch_factor=prefetch_factor,
            persistent_workers=persistent_workers,
        )
        self.iterator = None

    def init(self):
        self.iterator = iter(self)

    def fetch(self, **kwargs):
        if self.iterator is None:
            self.iterator = iter(self)

        try:
            return next(self.iterator), True
        except StopIteration:
            self.iterator = iter(self)
            return next(self.iterator), False
