#!/usr/bin/env python
"""Extract OpenFold pair representations from FASTA shards for SpatialPairSAE training."""

from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

from interplm.data_processing.pair_embedding_io import save_pair_shard
from interplm.embedders import get_embedder
from interplm.embedders.openfold_features import (
    build_predict_batch_from_a3m,
    iter_protein_inputs,
    read_fasta,
)


def extract_openfold_pairs_from_fasta(
    fasta_path: Path,
    output_dir: Path,
    *,
    msa_dir: Path | None = None,
    feature_dir: Path | None = None,
    model_name: str = "model_1",
    checkpoint_path: Path | None = None,
    layer: int = -1,
    max_msa_depth: int | None = None,
    max_proteins: int | None = None,
) -> dict[str, int]:
    """
    Extract pair representations for proteins in a FASTA shard and save to disk.

    Returns:
        Dictionary with counts of processed, skipped, and failed proteins.
    """
    if msa_dir is None and feature_dir is None:
        raise ValueError("Either msa_dir or feature_dir must be provided")

    embedder = get_embedder(
        "openfold",
        model_name=model_name,
        checkpoint_path=checkpoint_path,
    )

    pair_reprs: list[torch.Tensor] = []
    protein_ids: list[str] = []
    skipped = 0
    failed = 0

    fasta_records = read_fasta(fasta_path)
    if max_proteins is not None:
        fasta_records = fasta_records[:max_proteins]

    available_ids = {protein_id for protein_id, _ in fasta_records}

    for protein_id, sequence, model_input in tqdm(
        iter_protein_inputs(
            fasta_path,
            msa_dir=msa_dir,
            feature_dir=feature_dir,
            max_msa_depth=max_msa_depth,
        ),
        desc=f"Extracting {fasta_path.name}",
        unit="protein",
    ):
        if protein_id not in available_ids:
            continue
        try:
            if isinstance(model_input, tuple) and model_input[0] == "a3m":
                batch = build_predict_batch_from_a3m(
                    model_input[1],
                    model_name=model_name,
                    device=embedder.device,
                    max_msa_depth=max_msa_depth,
                )
            else:
                batch = model_input
            pair = embedder.embed(batch)
            if pair.shape[0] != 1:
                pair = pair[:1]
            pair_reprs.append(pair[0].detach().cpu())
            protein_ids.append(protein_id)
            if embedder.pair_dim is None:
                embedder.pair_dim = int(pair.shape[-1])
        except Exception as exc:
            failed += 1
            print(f"Failed {protein_id}: {exc}")
    skipped = len(available_ids) - len(protein_ids) - failed

    if not pair_reprs:
        raise RuntimeError(
            f"No pair representations extracted from {fasta_path}. "
            f"Check MSA/feature paths (skipped={skipped}, failed={failed})."
        )

    layer_dir = output_dir / f"layer_{layer}"
    shard_dir = layer_dir / fasta_path.stem
    save_pair_shard(
        shard_dir,
        pair_reprs,
        protein_ids,
        model=model_name,
        layer=layer,
    )

    print(
        f"Saved {len(pair_reprs)} pair representations to {shard_dir} "
        f"(skipped={skipped}, failed={failed})"
    )
    return {"processed": len(pair_reprs), "skipped": skipped, "failed": failed}


def main(
    fasta_dir: str,
    output_dir: str,
    msa_dir: Optional[str] = None,
    feature_dir: Optional[str] = None,
    model_name: str = "model_1",
    checkpoint_path: Optional[str] = None,
    layer: int = -1,
    max_msa_depth: Optional[int] = None,
    max_proteins: Optional[int] = None,
    shard_index: Optional[int] = None,
):
    fasta_dir = Path(fasta_dir)
    output_dir = Path(output_dir)
    msa_dir = Path(msa_dir) if msa_dir is not None else None
    feature_dir = Path(feature_dir) if feature_dir is not None else None
    checkpoint_path = Path(checkpoint_path) if checkpoint_path is not None else None
    """
    Extract OpenFold pair representations from FASTA shard files.

    Args:
        fasta_dir: Directory containing ``shard_*.fasta`` files.
        output_dir: Root output directory (creates ``layer_{layer}/shard_*``).
        msa_dir: Directory with per-protein A3M MSAs (OpenProteinSet layout).
        feature_dir: Optional directory with precomputed OpenFold feature batches.
        model_name: OpenFold config name (default: model_1).
        checkpoint_path: Optional OpenFold checkpoint path.
        layer: Layer label for output directory naming (default: -1 for final pair rep).
        max_msa_depth: Optional cap on MSA depth per protein.
        max_proteins: Optional cap on proteins per shard (useful for pilots).
        shard_index: Optional 0-based shard index to process a single FASTA file.
    """
    fasta_files = sorted(fasta_dir.glob("*.fasta"))
    if not fasta_files:
        fasta_files = sorted(fasta_dir.glob("*.fa"))
    if not fasta_files:
        raise FileNotFoundError(f"No FASTA files found in {fasta_dir}")

    if shard_index is not None:
        if shard_index < 0 or shard_index >= len(fasta_files):
            raise ValueError(
                f"Shard index {shard_index} out of range (0-{len(fasta_files) - 1})"
            )
        fasta_files = [fasta_files[shard_index]]
        print(f"Processing only shard {shard_index}: {fasta_files[0].name}")
    else:
        print(f"Found {len(fasta_files)} FASTA files")

    totals = {"processed": 0, "skipped": 0, "failed": 0}
    for fasta_file in fasta_files:
        counts = extract_openfold_pairs_from_fasta(
            fasta_file,
            output_dir,
            msa_dir=msa_dir,
            feature_dir=feature_dir,
            model_name=model_name,
            checkpoint_path=checkpoint_path,
            layer=layer,
            max_msa_depth=max_msa_depth,
            max_proteins=max_proteins,
        )
        for key in totals:
            totals[key] += counts[key]

    print(
        "Extraction complete: "
        f"processed={totals['processed']}, skipped={totals['skipped']}, failed={totals['failed']}"
    )
    print(f"Saved under {output_dir / f'layer_{layer}'}")


if __name__ == "__main__":
    from tap import tapify

    tapify(main)
