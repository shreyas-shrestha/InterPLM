"""Tests for pair embedding storage and loading."""

import torch

from interplm.data_processing.pair_embedding_io import (
    PAIR_ACTIVATION_TYPE,
    load_pair_shard,
    save_pair_shard,
)


def test_save_and_load_pair_shard(tmp_path):
    pair_reprs = [
        torch.randn(4, 4, 8),
        torch.randn(6, 6, 8),
    ]
    protein_ids = ["P11111", "P22222"]

    shard_dir = tmp_path / "shard_1"
    save_pair_shard(
        shard_dir,
        pair_reprs,
        protein_ids,
        model="model_1",
        layer=-1,
    )

    data = load_pair_shard(shard_dir)
    assert data["pair_reprs"].shape == (2, 6, 6, 8)
    assert data["seq_lengths"].tolist() == [4, 6]
    assert data["protein_ids"] == protein_ids

    first = data["pair_reprs"][0, :4, :4, :]
    assert torch.allclose(first, pair_reprs[0], atol=1e-6)

    metadata_path = shard_dir / "metadata.yaml"
    assert metadata_path.exists()
    text = metadata_path.read_text()
    assert PAIR_ACTIVATION_TYPE in text
