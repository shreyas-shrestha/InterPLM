"""Tests for pair-representation dataloader."""

import torch

from interplm.data_processing.pair_embedding_io import save_pair_shard
from interplm.train.pair_data_loader import PairActivationsDataLoader, PairDataloaderConfig


def test_pair_activations_dataloader_batches(tmp_path):
    shard_dir = tmp_path / "layer_-1" / "shard_1"
    save_pair_shard(
        shard_dir,
        [torch.randn(4, 4, 8), torch.randn(5, 5, 8), torch.randn(3, 3, 8)],
        ["A", "B", "C"],
        model="model_1",
        layer=-1,
    )

    loader = PairDataloaderConfig(
        plm_embd_dir=tmp_path / "layer_-1",
        batch_size=2,
        seed=0,
    ).build()

    assert loader.dataset.d_model == 8
    assert loader.dataset.total_tokens == 3

    batch = next(iter(loader))
    assert batch.ndim == 4
    assert batch.shape[0] == 2
    assert batch.shape[1] == batch.shape[2]
    assert batch.shape[-1] == 8
