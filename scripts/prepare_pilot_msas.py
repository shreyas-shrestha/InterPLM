#!/usr/bin/env python
"""Create single-sequence A3M files for pilot OpenFold extraction."""

from __future__ import annotations

from pathlib import Path

from interplm.embedders.openfold_features import read_fasta


def main(
    fasta_path: str,
    output_dir: str,
    max_proteins: int = 10,
):
    fasta_path = Path(fasta_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = read_fasta(fasta_path)[:max_proteins]

    for protein_id, sequence in records:
        a3m_path = output_dir / f"{protein_id}.a3m"
        with open(a3m_path, "w") as f:
            f.write(f">{protein_id}\n{sequence}\n")

    print(f"Wrote {len(records)} pilot MSAs to {output_dir}")


if __name__ == "__main__":
    from tap import tapify

    tapify(main)
