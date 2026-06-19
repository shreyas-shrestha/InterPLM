#!/usr/bin/env python
"""
Train a SpatialPairSAE on OpenFold pair representations.

Usage:
    export INTERPLM_DATA=/path/to/data
    export LAYER=-1
    python examples/train_spatial_pair_sae.py
"""

from __future__ import annotations

import os
from pathlib import Path

from interplm.train.checkpoint_manager import CheckpointConfig
from interplm.train.configs import TrainingRunConfig
from interplm.train.evaluation import EvaluationConfig
from interplm.train.pair_data_loader import PairDataloaderConfig
from interplm.train.training_run import SAETrainingRun
from interplm.train.trainers import SpatialPairTrainerConfig
from interplm.train.wandb_manager import WandbConfig


def main() -> int:
    INTERPLM_DATA = Path(os.environ.get("INTERPLM_DATA", "data"))
    LAYER = os.environ.get("LAYER", "-1")

    embeddings_dir = INTERPLM_DATA / "training_embeddings" / "openfold_baseline" / f"layer_{LAYER}"
    eval_embeddings_dir = INTERPLM_DATA / "eval_embeddings" / "openfold_baseline" / f"layer_{LAYER}"
    save_dir = Path("models") / "openfold_spatial_pair_sae" / f"layer_{LAYER}"

    batch_size = int(os.environ.get("PAIR_BATCH_SIZE", "4"))
    steps = int(os.environ.get("TRAIN_STEPS", "50000"))
    expansion_factor = int(os.environ.get("EXPANSION_FACTOR", "4"))
    learning_rate = float(os.environ.get("LEARNING_RATE", "2e-4"))
    l1_penalty = float(os.environ.get("L1_PENALTY", "0.06"))

    print("=" * 60)
    print("SpatialPairSAE Training")
    print("=" * 60)
    print(f"Training embeddings: {embeddings_dir}")
    print(f"Evaluation embeddings: {eval_embeddings_dir}")
    print(f"Save directory: {save_dir}")
    print(f"Batch size (proteins): {batch_size}")
    print(f"Expansion factor: {expansion_factor}")
    print()

    dataloader_cfg = PairDataloaderConfig(
        plm_embd_dir=embeddings_dir,
        batch_size=batch_size,
    )

    trainer_cfg = SpatialPairTrainerConfig(
        expansion_factor=expansion_factor,
        lr=learning_rate,
        l1_penalty=l1_penalty,
        warmup_steps=max(1, int(steps * 0.05)),
        decay_start=max(1, int(steps * 0.8)),
        steps=steps,
        normalize_to_sqrt_d=False,
    )

    eval_cfg = EvaluationConfig(
        eval_embd_dir=eval_embeddings_dir if eval_embeddings_dir.exists() else None,
        eval_steps=5000,
        eval_batch_size=batch_size,
        activation_type="pair",
    )

    wandb_cfg = WandbConfig(use_wandb=False)
    checkpoint_cfg = CheckpointConfig(
        save_dir=save_dir,
        save_steps=5000,
        max_ckpts_to_keep=2,
    )

    config = TrainingRunConfig(
        dataloader_cfg=dataloader_cfg,
        trainer_cfg=trainer_cfg,
        eval_cfg=eval_cfg,
        wandb_cfg=wandb_cfg,
        checkpoint_cfg=checkpoint_cfg,
    )

    training_run = SAETrainingRun.from_config(config)
    training_run.run()

    print()
    print("=" * 60)
    print(f"Training complete. Model saved to {save_dir / 'ae.pt'}")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
