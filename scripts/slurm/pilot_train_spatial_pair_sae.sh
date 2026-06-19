#!/bin/bash
#SBATCH --job-name=pilot-pair-sae
#SBATCH --account=bekh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=/u/sshrestha9/interPLM/logs/pilot_train_%j.out
#SBATCH --error=/u/sshrestha9/interPLM/logs/pilot_train_%j.err

set -euo pipefail

PYTHON=/sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/bin/python
PIP=/sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/bin/pip

export INTERPLM_DATA=/u/sshrestha9/interplm_data
export LAYER=-1
export EMBEDDINGS_SUBDIR=openfold_pilot
export SAVE_SUBDIR=openfold_pilot_sae
export PAIR_BATCH_SIZE=1
export TRAIN_STEPS=500
export EXPANSION_FACTOR=4
export L1_PENALTY=0.06

cd /u/sshrestha9/interPLM

echo "=== Pilot SpatialPairSAE training ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi || true

$PIP install -e /u/sshrestha9/interPLM -q

EMBEDDINGS_DIR="${INTERPLM_DATA}/training_embeddings/${EMBEDDINGS_SUBDIR}/layer_${LAYER}/shard_1"
if [[ ! -f "${EMBEDDINGS_DIR}/activations.pt" ]]; then
  echo "Missing pilot embeddings at ${EMBEDDINGS_DIR}"
  exit 1
fi

$PYTHON examples/train_spatial_pair_sae.py

echo "=== Pilot training complete ==="
ls -la "models/${SAVE_SUBDIR}/layer_${LAYER}/" || true
