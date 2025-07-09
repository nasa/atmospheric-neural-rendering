from collections.abc import Iterator

import torch
from torch.utils.data.sampler import BatchSampler, RandomSampler, SequentialSampler

from atmonr.datasets.factory import Dataset, ExtractDataset


class BatchLoader:
    """Batch-wise dataloader."""

    def __init__(
        self,
        dataset: Dataset | ExtractDataset,
        batch_size: int,
        shuffle=True,
        drop_last: bool = False,
    ):
        """Initialize a BatchLoader.

        Args:
            dataset: A Dataset or ExtractDataset. Must have a __getbatch__ function.
            batch_size: Number of samples per batch.
            shuffle: Whether to shuffle the sample index (default: True).
            drop_last: Whether to drop the last batch if it's smaller than batch size
                (default: False).
        """
        self.dataset = dataset
        if isinstance(dataset, Dataset):
            self.idx = dataset.ray_idx
        else:
            self.idx = dataset.idx

        if shuffle:
            inst_sampler = RandomSampler(self.idx)
        else:
            inst_sampler = SequentialSampler(self.idx)

        self.sampler = BatchSampler(
            inst_sampler,
            batch_size=batch_size,
            drop_last=drop_last,
        )

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        for idx in self.sampler:
            idx = torch.LongTensor(idx).to(self.idx.device)
            batch = self.dataset.__getbatch__(idx)
            yield batch

    def __len__(self) -> int:
        return len(self.sampler)
