"""
Data loader for sharded OpenFold pair representations.

Each dataset sample is one protein's pair matrix with shape ``[N, N, d_hidden]``.
Batches are padded to the longest sequence in the batch and returned as
``[batch, N_max, N_max, d_hidden]``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from interplm.data_processing.pair_embedding_io import (
    PAIR_ACTIVATION_TYPE,
    load_pair_shard,
    load_pair_shard_metadata,
)
from interplm.utils import get_device
from interplm.train.data_loader import AttributePreservingSubset


@dataclass
class PairDataloaderConfig:
    """Configuration for pair-representation SAE training."""

    plm_embd_dir: Path
    batch_size: int = 4
    seed: int = 0
    samples_to_skip: int = 0
    n_shards_to_include: int | None = None
    target_dtype: torch.dtype = torch.float32
    activation_type: str = "pair"

    def build(self) -> "PairActivationsDataLoader":
        return PairActivationsDataLoader(self)


class PairShardDataset(Dataset):
    """One shard of padded pair representations indexed by protein."""

    def __init__(
        self,
        shard_dir: Path,
        shuffle: bool = True,
        seed: int | None = None,
    ):
        self.shard_dir = shard_dir
        self.metadata = load_pair_shard_metadata(shard_dir)
        if self.metadata is None:
            raise FileNotFoundError(f"Missing metadata.yaml in {shard_dir}")

        self.d_model = int(self.metadata["d_model"])
        self.total_proteins = int(self.metadata["total_proteins"])
        self.total_tokens = self.total_proteins
        self.layer = self.metadata.get("layer")
        self.max_seq_len = int(self.metadata.get("max_seq_len", 0))

        if seed is not None:
            torch.manual_seed(seed)

        self._data = None
        self._pair_reprs = None
        self._seq_lengths = None
        self._permutation = None
        self.shuffle = shuffle

    def _ensure_loaded(self) -> None:
        if self._data is not None:
            return
        self._data = load_pair_shard(self.shard_dir, map_location="cpu")
        self._pair_reprs = self._data["pair_reprs"]
        self._seq_lengths = self._data["seq_lengths"]

    @property
    def permutation(self) -> torch.Tensor:
        self._ensure_loaded()
        if self._permutation is None:
            self._permutation = (
                torch.randperm(self.total_proteins)
                if self.shuffle
                else torch.arange(self.total_proteins)
            )
        return self._permutation

    def __len__(self) -> int:
        return self.total_proteins

    def __getitem__(self, idx: int) -> torch.Tensor:
        self._ensure_loaded()
        protein_idx = int(self.permutation[idx].item())
        length = int(self._seq_lengths[protein_idx].item())
        pair = self._pair_reprs[protein_idx, :length, :length, :]
        return pair.to(dtype=torch.float32)


class ShardedPairActivationsDataset(Dataset):
    """Concatenated pair shards under ``layer_*/shard_*`` directories."""

    def __init__(
        self,
        root_dir: str | Path,
        shuffle: bool = True,
        seed: int | None = None,
        n_shards_to_include: int | None = None,
    ):
        self.root_dir = Path(root_dir)
        self.datasets: list[dict] = []
        self.total_tokens = 0
        self.total_proteins = 0
        self.d_model = None
        self.cumulative_proteins = [0]
        self.shuffle = shuffle

        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        shard_dirs = sorted(
            d
            for d in self.root_dir.iterdir()
            if d.is_dir() and (d / "metadata.yaml").exists()
        )
        if not shard_dirs:
            shard_dirs = sorted(
                d
                for d in self.root_dir.glob("**/shard_*")
                if d.is_dir() and (d / "metadata.yaml").exists()
            )

        if n_shards_to_include is not None:
            shard_dirs = shard_dirs[:n_shards_to_include]

        print(f"Loading pair metadata from {len(shard_dirs)} shard directories...")
        for shard_dir in tqdm(shard_dirs):
            metadata = load_pair_shard_metadata(shard_dir)
            if metadata is None:
                continue
            if metadata.get("activation_type") not in (None, PAIR_ACTIVATION_TYPE):
                print(f"Skipping {shard_dir} - activation_type={metadata.get('activation_type')}")
                continue

            dataset = PairShardDataset(shard_dir, shuffle=shuffle, seed=seed)
            info = {
                "total_proteins": dataset.total_proteins,
                "total_tokens": dataset.total_proteins,
                "d_model": dataset.d_model,
                "dataset": dataset,
                "layer": dataset.layer,
            }
            self.datasets.append(info)
            self.total_proteins += dataset.total_proteins
            self.total_tokens = self.total_proteins
            self.cumulative_proteins.append(self.total_proteins)
            if self.d_model is None:
                self.d_model = dataset.d_model
            else:
                assert self.d_model == dataset.d_model

    def __len__(self) -> int:
        return self.total_proteins

    def __getitem__(self, idx: int) -> torch.Tensor:
        shard_index = (
            next(i for i, cum in enumerate(self.cumulative_proteins) if cum > idx) - 1
        )
        local_idx = idx - self.cumulative_proteins[shard_index]
        return self.datasets[shard_index]["dataset"][local_idx]


def collate_pair_batch(batch: list[torch.Tensor]) -> torch.Tensor:
    max_len = max(item.shape[0] for item in batch)
    d_hidden = batch[0].shape[-1]
    out = torch.zeros(len(batch), max_len, max_len, d_hidden, dtype=batch[0].dtype)
    for i, item in enumerate(batch):
        length = item.shape[0]
        out[i, :length, :length, :] = item
    return out



class PairActivationsDataLoader(DataLoader):
    def __init__(self, dataloader_config: PairDataloaderConfig):
        self.config = dataloader_config

        if self.config.seed is not None:
            torch.manual_seed(self.config.seed)
            torch.cuda.manual_seed_all(self.config.seed)

        dataset: Dataset = ShardedPairActivationsDataset(
            self.config.plm_embd_dir,
            seed=self.config.seed,
            shuffle=True,
            n_shards_to_include=self.config.n_shards_to_include,
        )

        if self.config.samples_to_skip > 0:
            dataset = AttributePreservingSubset(
                dataset,
                range(self.config.samples_to_skip, len(dataset)),
            )

        device = get_device()

        def collate_fn(batch: list[torch.Tensor]) -> torch.Tensor:
            return collate_pair_batch(batch).to(device=device, dtype=self.config.target_dtype)

        super().__init__(
            dataset,
            batch_size=dataloader_config.batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )
