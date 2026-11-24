#!/bin/bash
# CPU job (test): recompute the arithmetic slice summaries from the stored harvests (no GPU).
set -euo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path"); A=$WORK/arith_v1
OUT=outputs/arith_summary_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
for split in discovery heldout; do
  $PY scripts/gates/arith_summary.py --harvest "$A/harvest_$split" --prompts "$A/$split" --truth "$A/ground_truth.json" \
      --out "$OUT/summary_$split.json" --model "$SNAP"
done
cp .code_commit "$OUT/" 2>/dev/null || true
echo SUMMARY_OK
