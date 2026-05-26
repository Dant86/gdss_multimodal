#!/usr/bin/env bash
# SLURM job: ECGUNet autoencoder pre-training.
#
# Reads cluster config from .env in the project root.
# A100 80 GB target: batch_size=512, ~20 min for 30 K steps.
#
# Submit: sbatch scripts/pretrain.sh
# Override steps:  sbatch scripts/pretrain.sh --max-steps 50000
# Resume:          sbatch scripts/pretrain.sh --resume checkpoints/pretrain_step_0020000.pt
#
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we start with /dev/null and exec-redirect once LOG_DIR is known.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_pretrain
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

set -euo pipefail

export PATH="$HOME/.local/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$SLURM_SUBMIT_DIR"

if [[ ! -f .env ]]; then
    echo "ERROR: .env not found in ${SLURM_SUBMIT_DIR}. Copy .env.sample and fill in your values." >&2
    exit 1
fi
source .env

[[ -n "${SLURM_PARTITION:-}" ]] && SBATCH_PARTITION="$SLURM_PARTITION"
[[ -n "${SLURM_ACCOUNT:-}"   ]] && SBATCH_ACCOUNT="$SLURM_ACCOUNT"

mkdir -p "${LOG_DIR}"
exec > "${LOG_DIR}/pretrain_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/pretrain_${SLURM_JOB_ID}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

# ── Force PyTorch's bundled cuDNN to take precedence over system cuDNN ────────
TORCH_LIB="$(python -c 'import torch, pathlib; print(pathlib.Path(torch.__file__).parent / "lib")')"
export LD_LIBRARY_PATH="${TORCH_LIB}:${LD_LIBRARY_PATH:-}"


# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting pretrain (job ${SLURM_JOB_ID})"
echo "DATA_DIR       = ${DATA_DIR}"
echo "CACHE_DIR      = ${CACHE_DIR}"
echo "CHECKPOINT_DIR = ${CHECKPOINT_DIR}"
echo "LOG_DIR        = ${LOG_DIR}"

python apps/pretrain/main.py \
    --data-dir        "${DATA_DIR}" \
    --cache-dir       "${CACHE_DIR}" \
    --checkpoint-dir  "${CHECKPOINT_DIR}" \
    --batch-size      512 \
    --max-steps       15000 \
    --lr              1e-3 \
    --spectral-weight 0.1 \
    --device          cuda \
    train.num_workers=0 \
    "$@"

echo "[$(date)] Done."
