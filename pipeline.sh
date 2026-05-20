#!/usr/bin/env bash
# pipeline.sh — sequential Modal pipeline:
#   1. Wait for current training to finish
#   2. Sample + visualize current 12-lead model  → figures/12lead_v1/
#   3. Pretrain new model (12-lead + CFG + loss clamp)
#   4. DSM train new model
#   5. Sample + visualize new model              → figures/12lead_v2_cfg/
#
# Usage: bash pipeline.sh <training_log_file>
#
# Example:
#   bash pipeline.sh /tmp/.../tasks/bk5rw3yrz.output

set -euo pipefail

TRAIN_LOG="${1:?Usage: bash pipeline.sh <training_log_file>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/modal"

echo "=== Pipeline started at $(date) ==="
echo "    Watching: $TRAIN_LOG"

# ── Step 1: wait for current training ──────────────────────────────────────────
echo ""
echo ">>> Waiting for current training run to complete…"
until grep -q "Training complete\." "$TRAIN_LOG" 2>/dev/null; do
    sleep 30
done
echo "    Training complete at $(date)"

# ── Step 2: sample + visualize current model ───────────────────────────────────
echo ""
echo ">>> [v1] Generating 1000 samples (Lead II, s4, NFE=1000)…"
"$VENV" run "$SCRIPT_DIR/sample.py" \
    --checkpoint final \
    --sampler s4 \
    --n-steps 1000 \
    --n-samples 1000 \
    --batch-size 32

echo ""
echo ">>> [v1] Generating figures → figures/12lead_v1/"
"$VENV" run "$SCRIPT_DIR/visualize.py" \
    --checkpoint final \
    --n-gen 300 \
    --tag 12lead_v1

# ── Step 3: pretrain new model ─────────────────────────────────────────────────
echo ""
echo ">>> [v2] Pretraining new ECGUNet (12-lead + CFG)…"
"$VENV" run "$SCRIPT_DIR/pretrain.py"

# ── Step 4: DSM train new model ────────────────────────────────────────────────
echo ""
echo ">>> [v2] DSM training (12-lead + CFG + loss clamp)…"
"$VENV" run "$SCRIPT_DIR/train.py" \
    --pretrain-checkpoint /vol/checkpoints/pretrain_final.pt \
    --max-steps 100000

# ── Step 5: sample + visualize new model ───────────────────────────────────────
echo ""
echo ">>> [v2] Generating 1000 samples (Lead II, s4, NFE=1000, cfg_scale=1.5)…"
"$VENV" run "$SCRIPT_DIR/sample.py" \
    --checkpoint final \
    --sampler s4 \
    --n-steps 1000 \
    --n-samples 1000 \
    --batch-size 32 \
    --cfg-scale 1.5

echo ""
echo ">>> [v2] Generating figures → figures/12lead_v2_cfg/"
"$VENV" run "$SCRIPT_DIR/visualize.py" \
    --checkpoint final \
    --n-gen 300 \
    --tag 12lead_v2_cfg

echo ""
echo "=== Pipeline complete at $(date) ==="
echo "    figures/12lead_v1/  — 12-lead model (no CFG)"
echo "    figures/12lead_v2_cfg/ — 12-lead model + CFG (scale=1.5)"
