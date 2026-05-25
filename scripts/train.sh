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
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_train
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1

set -euo pipefail

# ── Load environment ──────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
ENV_FILE="$PROJECT_DIR/.env"

if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

[[ -n "${SLURM_PARTITION:-}" ]] && SBATCH_PARTITION="$SLURM_PARTITION"
[[ -n "${SLURM_ACCOUNT:-}"   ]] && SBATCH_ACCOUNT="$SLURM_ACCOUNT"

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_DIR="${LOG_DIR:-$PROJECT_DIR/logs}"
mkdir -p "$LOG_DIR"
exec >"$LOG_DIR/train_${SLURM_JOB_ID:-local}.out" \
    2>"$LOG_DIR/train_${SLURM_JOB_ID:-local}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
source "$VENV_DIR/bin/activate"

# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

echo "=== train  $(date) ==="
echo "DATA_DIR        = ${DATA_DIR:-data/ptbxl}"
echo "CACHE_DIR       = ${CACHE_DIR:-cache}"
echo "CHECKPOINT_DIR  = ${CHECKPOINT_DIR:-checkpoints}"
echo "LOG_DIR         = $LOG_DIR"

# Extra CLI args forwarded from sbatch (e.g. --pretrain-checkpoint, --resume-checkpoint)
EXTRA_ARGS="${@:-}"

python apps/train/main.py \
    --data-dir        "${DATA_DIR:-data/ptbxl}" \
    --cache-dir       "${CACHE_DIR:-cache}" \
    --checkpoint-dir  "${CHECKPOINT_DIR:-checkpoints}" \
    --batch-size      256 \
    --max-steps       100000 \
    --lr              3e-4 \
    --device          cuda \
    $EXTRA_ARGS

echo "=== done $(date) ==="
