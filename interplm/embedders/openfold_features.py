"""Utilities for building OpenFold inputs from FASTA files and MSAs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterator

import torch

# OpenFold/AlphaFold restype order (21 classes including X).
RESTYPES = "ARNDCQEGHILKMFPSTWYXX"
RESTYPE_TO_INDEX = {aa: idx for idx, aa in enumerate(RESTYPES[:21])}
RESTYPE_TO_INDEX["-"] = RESTYPE_TO_INDEX["X"]
RESTYPE_TO_INDEX["."] = RESTYPE_TO_INDEX["X"]
RESTYPE_TO_INDEX["U"] = RESTYPE_TO_INDEX["C"]
RESTYPE_TO_INDEX["B"] = RESTYPE_TO_INDEX["D"]
RESTYPE_TO_INDEX["Z"] = RESTYPE_TO_INDEX["E"]
RESTYPE_TO_INDEX["J"] = RESTYPE_TO_INDEX["X"]
RESTYPE_TO_INDEX["O"] = RESTYPE_TO_INDEX["X"]
RESTYPE_TO_INDEX["*"] = RESTYPE_TO_INDEX["X"]
RESTYPE_TO_INDEX["~"] = RESTYPE_TO_INDEX["X"]
for aa in RESTYPES[:20]:
    RESTYPE_TO_INDEX[aa.lower()] = RESTYPE_TO_INDEX[aa]


def parse_uniprot_id(header: str) -> str:
    """Extract a UniProt accession from a FASTA header."""
    header = header.lstrip(">").strip()
    if "|" in header:
        parts = header.split("|")
        if len(parts) >= 2 and parts[1]:
            return parts[1]
    token = header.split()[0]
    token = re.sub(r"[^A-Za-z0-9_-]", "", token)
    return token


def read_fasta(path: Path) -> list[tuple[str, str]]:
    """Return ``[(protein_id, sequence), ...]`` from a FASTA file."""
    records: list[tuple[str, str]] = []
    header: str | None = None
    seq_parts: list[str] = []

    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((parse_uniprot_id(header), "".join(seq_parts)))
                header = line[1:]
                seq_parts = []
            else:
                seq_parts.append(line)

    if header is not None:
        records.append((parse_uniprot_id(header), "".join(seq_parts)))

    return records


def _a3m_row_to_aligned(row: str) -> str:
    """Convert one A3M row to an aligned sequence (drop insertions, keep gaps)."""
    aligned = []
    for char in row:
        if char.islower():
            continue
        aligned.append(char.upper())
    return "".join(aligned)


def read_a3m_sequences(path: Path) -> list[str]:
    """Read aligned sequences from an A3M file."""
    sequences: list[str] = []
    current: list[str] = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append(_a3m_row_to_aligned("".join(current)))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append(_a3m_row_to_aligned("".join(current)))
    return sequences


def sequence_to_indices(sequence: str) -> torch.Tensor:
    indices = []
    for aa in sequence.upper():
        indices.append(RESTYPE_TO_INDEX.get(aa, RESTYPE_TO_INDEX["X"]))
    return torch.tensor(indices, dtype=torch.long)


def a3m_to_msa_tensor(path: Path, max_msa_depth: int | None = None) -> torch.Tensor:
    """
    Convert an A3M alignment file to an integer MSA tensor ``[1, M, N]``.

    Gaps in the query row are removed so ``N`` matches the ungapped query length.
    Other MSA rows are converted at the same column positions.
    """
    sequences = read_a3m_sequences(path)
    if not sequences:
        raise ValueError(f"No sequences found in A3M file: {path}")

    if max_msa_depth is not None:
        sequences = sequences[:max_msa_depth]

    aligned_length = len(sequences[0])
    for seq in sequences[1:]:
        if len(seq) != aligned_length:
            raise ValueError(
                f"MSA sequences in {path} have inconsistent aligned lengths "
                f"({aligned_length} vs {len(seq)})"
            )

    query = sequences[0]
    keep_columns = [idx for idx, aa in enumerate(query) if aa != "-"]
    seq_len = len(keep_columns)
    if seq_len == 0:
        raise ValueError(f"Query sequence in {path} is empty after removing gaps")

    msa_rows = []
    for seq in sequences:
        filtered = "".join(seq[idx] for idx in keep_columns)
        msa_rows.append(sequence_to_indices(filtered))

    msa = torch.stack(msa_rows, dim=0).unsqueeze(0)
    return msa


def resolve_msa_path(msa_dir: Path, protein_id: str) -> Path | None:
    """Find an MSA file for a UniProt accession using common naming conventions."""
    candidates = [
        msa_dir / f"{protein_id}.a3m",
        msa_dir / protein_id / "msa.a3m",
        msa_dir / protein_id / f"{protein_id}.a3m",
        msa_dir / f"{protein_id}.sto",
        msa_dir / protein_id / "msa.sto",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_feature_path(feature_dir: Path, protein_id: str) -> Path | None:
    """Find a precomputed OpenFold feature batch for a protein."""
    candidates = [
        feature_dir / f"{protein_id}.pt",
        feature_dir / f"{protein_id}.pkl",
        feature_dir / protein_id / "features.pt",
        feature_dir / protein_id / "features.pkl",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_feature_batch(path: Path, map_location: str | torch.device = "cpu") -> dict:
    data = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(data, dict):
        return data
    raise TypeError(f"Expected feature dict at {path}, got {type(data)}")


def iter_protein_inputs(
    fasta_path: Path,
    *,
    msa_dir: Path | None = None,
    feature_dir: Path | None = None,
    max_msa_depth: int | None = None,
) -> Iterator[tuple[str, str, dict | torch.Tensor]]:
    """
    Yield ``(protein_id, sequence, msa_or_feature_batch)`` for each FASTA record.

    Prefers precomputed OpenFold feature batches when ``feature_dir`` is set.
    Otherwise looks up A3M MSAs under ``msa_dir``.
    """
    if msa_dir is None and feature_dir is None:
        raise ValueError("Either msa_dir or feature_dir must be provided")

    for protein_id, sequence in read_fasta(fasta_path):
        if feature_dir is not None:
            feature_path = resolve_feature_path(feature_dir, protein_id)
            if feature_path is not None:
                yield protein_id, sequence, load_feature_batch(feature_path)
                continue

        if msa_dir is None:
            continue

        msa_path = resolve_msa_path(msa_dir, protein_id)
        if msa_path is None:
            continue

        if msa_path.suffix == ".a3m":
            yield protein_id, sequence, a3m_to_msa_tensor(msa_path, max_msa_depth=max_msa_depth)
        else:
            raise ValueError(
                f"Unsupported MSA format for {msa_path}. Use .a3m or precomputed feature batches."
            )
