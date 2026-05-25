#!/usr/bin/env bash
# SLURM job: ECGUNet autoencoder pre-training.
#
# Reads cluster config from .env in the project root.
# A100 80 GB target: batch_size=512, ~20 min for 30 K steps.
#
# Submit: sbatch scripts/pretrain.sh
# Override steps:  sbatch scripts/pretrain.sh --max-steps 50000
# Resume:          sbatch scripts/pretrain.sh --resume checkpoints/pretrain_step_0020000.pt
# ──────────────────────────────────────────────────────────────────────────────

#SBATCH --job-name=gdss_pretrain
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
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
exec >"$LOG_DIR/pretrain_${SLURM_JOB_ID:-local}.out" \
    2>"$LOG_DIR/pretrain_${SLURM_JOB_ID:-local}.err"

# ── Activate virtualenv ───────────────────────────────────────────────────────
VENV_DIR="${VENV_DIR:-$PROJECT_DIR/.venv}"
source "$VENV_DIR/bin/activate"

# ── GPU diagnostics ───────────────────────────────────────────────────────────
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# ── Run ───────────────────────────────────────────────────────────────────────
cd "$PROJECT_DIR"

echo "=== pretrain  $(date) ==="
echo "DATA_DIR        = ${DATA_DIR:-data/ptbxl}"
echo "CACHE_DIR       = ${CACHE_DIR:-cache}"
echo "CHECKPOINT_DIR  = ${CHECKPOINT_DIR:-checkpoints}"
echo "LOG_DIR         = $LOG_DIR"

# Extra CLI args forwarded from sbatch (e.g. --resume, --max-steps)
EXTRA_ARGS="${@:-}"

python apps/pretrain/main.py \
    --data-dir        "${DATA_DIR:-data/ptbxl}" \
    --cache-dir       "${CACHE_DIR:-cache}" \
    --checkpoint-dir  "${CHECKPOINT_DIR:-checkpoints}" \
    --batch-size      512 \
    --max-steps       30000 \
    --lr              1e-3 \
    --spectral-weight 0.1 \
    --device          cuda \
    $EXTRA_ARGS

echo "=== done $(date) ==="
