#!/bin/bash
#SBATCH --job-name=pilot-of-extract
#SBATCH --account=bekh-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:nvidia_a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/u/sshrestha9/interPLM/logs/pilot_extract_%j.out
#SBATCH --error=/u/sshrestha9/interPLM/logs/pilot_extract_%j.err

set -euo pipefail

PYTHON=/sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/bin/python
PIP=/sw/rh9.4/user/python/conda-env/pytorch-2.8-cu128/bin/pip

export INTERPLM_DATA=/u/sshrestha9/interplm_data
export LAYER=-1
export OPENFOLD_ROOT=/u/sshrestha9/openfold
export PYTHONPATH="${OPENFOLD_ROOT}/build/lib.linux-x86_64-cpython-311:${OPENFOLD_ROOT}:${PYTHONPATH:-}"

cd /u/sshrestha9/interPLM

echo "=== Pilot OpenFold pair extraction ==="
echo "Host: $(hostname)"
echo "Date: $(date)"
nvidia-smi || true

echo "=== Installing dependencies ==="
$PIP install -e /u/sshrestha9/interPLM -q
$PIP install ml-collections dm-tree biopython scipy modelcif dm-haiku -q
$PYTHON -c "from openfold.model.model import AlphaFold; print('OpenFold import OK')"

echo "=== Ensuring OpenFold checkpoint ==="
PARAMS_ROOT="${INTERPLM_DATA}/openfold_params"
PARAMS_DIR="${PARAMS_ROOT}/openfold_params"
mkdir -p "${PARAMS_ROOT}"
if [[ ! -d "${PARAMS_DIR}" ]]; then
  echo "Downloading OpenFold params from HuggingFace..."
  bash "${OPENFOLD_ROOT}/scripts/download_openfold_params_huggingface.sh" "${PARAMS_ROOT}"
fi
# HuggingFace clone stores LFS pointer stubs unless pulled.
CHECKPOINT_PATH="${PARAMS_DIR}/finetuning_no_templ_2.pt"
if [[ ! -f "${CHECKPOINT_PATH}" ]] || [[ "$(stat -c%s "${CHECKPOINT_PATH}")" -lt 1000000 ]]; then
  echo "Downloading OpenFold checkpoint via HuggingFace Hub..."
  $PIP install huggingface_hub -q
  CHECKPOINT_PATH="$($PYTHON - <<'PY'
from huggingface_hub import hf_hub_download
from pathlib import Path
import os
dest_dir = Path(os.environ["INTERPLM_DATA"]) / "openfold_params" / "weights"
dest_dir.mkdir(parents=True, exist_ok=True)
path = hf_hub_download(
    repo_id="nz/OpenFold",
    filename="finetuning_no_templ_2.pt",
    local_dir=str(dest_dir),
)
print(path)
PY
)"
fi
echo "Using checkpoint: ${CHECKPOINT_PATH}"

MSA_DIR="${INTERPLM_DATA}/pilot_msas/shard_1"
OUTPUT_DIR="${INTERPLM_DATA}/training_embeddings/openfold_pilot"

echo "=== Running extraction on 10 proteins from shard_1 ==="
$PYTHON scripts/extract_openfold_pairs.py \
  --fasta_dir "${INTERPLM_DATA}/uniprot_shards" \
  --output_dir "${OUTPUT_DIR}" \
  --msa_dir "${MSA_DIR}" \
  --checkpoint_path "${CHECKPOINT_PATH}" \
  --model_name model_3 \
  --layer "${LAYER}" \
  --shard_index 0 \
  --max_proteins 10 \
  --max_msa_depth 64

echo "=== Pilot extraction complete ==="
ls -la "${OUTPUT_DIR}/layer_${LAYER}/shard_1/"
