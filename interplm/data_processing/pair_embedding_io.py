"""Save and load sharded OpenFold pair-representation activations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
import yaml

PAIR_ACTIVATION_TYPE = "pair"


def save_pair_shard(
    output_dir: Path,
    pair_reprs: list[torch.Tensor],
    protein_ids: list[str],
    *,
    model: str,
    layer: int,
    dtype: str = "float32",
) -> None:
    """
    Save one shard of pair representations for SpatialPairSAE training.

    Each entry in ``pair_reprs`` has shape ``[N, N, d_hidden]``. Stored tensors
    are padded to the max sequence length within the shard.
    """
    if len(pair_reprs) != len(protein_ids):
        raise ValueError("pair_reprs and protein_ids must have the same length")
    if not pair_reprs:
        raise ValueError("Cannot save an empty pair shard")

    output_dir.mkdir(parents=True, exist_ok=True)

    seq_lengths = torch.tensor([x.shape[0] for x in pair_reprs], dtype=torch.long)
    d_hidden = pair_reprs[0].shape[-1]
    max_len = int(seq_lengths.max().item())

    padded = torch.zeros(len(pair_reprs), max_len, max_len, d_hidden, dtype=torch.float32)
    for i, pair in enumerate(pair_reprs):
        length = pair.shape[0]
        if pair.shape != (length, length, d_hidden):
            raise ValueError(
                f"Protein {protein_ids[i]}: expected square [N, N, {d_hidden}], got {tuple(pair.shape)}"
            )
        padded[i, :length, :length, :] = pair.float()

    torch.save(
        {
            "pair_reprs": padded,
            "seq_lengths": seq_lengths,
            "protein_ids": protein_ids,
        },
        output_dir / "activations.pt",
    )

    metadata = {
        "model": model,
        "layer": layer,
        "d_model": int(d_hidden),
        "total_proteins": len(pair_reprs),
        "total_tokens": len(pair_reprs),
        "max_seq_len": max_len,
        "dtype": dtype,
        "activation_type": PAIR_ACTIVATION_TYPE,
    }
    with open(output_dir / "metadata.yaml", "w") as f:
        yaml.dump(metadata, f, default_flow_style=False)


def load_pair_shard_metadata(shard_dir: Path) -> dict[str, Any] | None:
    metadata_path = shard_dir / "metadata.yaml"
    if not metadata_path.exists():
        return None
    with open(metadata_path, "r") as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def load_pair_shard(shard_dir: Path, map_location: str | torch.device = "cpu") -> dict[str, Any]:
    activations_path = shard_dir / "activations.pt"
    if not activations_path.exists():
        raise FileNotFoundError(f"Missing pair activations file: {activations_path}")

    data = torch.load(activations_path, map_location=map_location, weights_only=False)
    required = {"pair_reprs", "seq_lengths", "protein_ids"}
    missing = required - set(data.keys())
    if missing:
        raise KeyError(f"Pair shard at {activations_path} missing keys: {sorted(missing)}")
    return data
