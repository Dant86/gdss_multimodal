#!/usr/bin/env bash
# SLURM job: joint ECG–text diffusion model training.
#
# Reads cluster config from .env in the project root.
# A100 80 GB target: batch_size=256 for 12-lead multi-lead mode.
#
# Submit: sbatch scripts/train.sh
# With pretrain weights:
#   sbatch scripts/train.sh --pretrain-checkpoint checkpoints/pretrain_final.pt
# Resume:
#   sbatch scripts/train.sh --resume-checkpoint checkpoints/step_0050000.pt
#
# Log paths are set after sourcing .env — SLURM directives can't read env files,
# so we start with /dev/null and exec-redirect once LOG_DIR is known.
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_train
#SBATCH --partition=general
#SBATCH --qos=general
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
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
exec > "${LOG_DIR}/train_${SLURM_JOB_ID}.out" \
     2> "${LOG_DIR}/train_${SLURM_JOB_ID}.err"

# ── Pin CUDA 12.4 (highest version available on this cluster) ─────────────────
export CUDA_HOME=/usr/local/cuda-12.4
export LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH:-}"
export PATH="${CUDA_HOME}/bin:${PATH}"

# ── Activate virtualenv ───────────────────────────────────────────────────────
source "${VENV_DIR}/bin/activate"

# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
echo "[$(date)] Starting train (job ${SLURM_JOB_ID})"
echo "DATA_DIR       = ${DATA_DIR}"
echo "CACHE_DIR      = ${CACHE_DIR}"
echo "CHECKPOINT_DIR = ${CHECKPOINT_DIR}"
echo "LOG_DIR        = ${LOG_DIR}"

python apps/train/main.py \
    --data-dir        "${DATA_DIR}" \
    --cache-dir       "${CACHE_DIR}" \
    --checkpoint-dir  "${CHECKPOINT_DIR}" \
    --batch-size      256 \
    --max-steps       100000 \
    --lr              3e-4 \
    --device          cuda \
    train.num_workers=0 \
    "$@"

echo "[$(date)] Done."
