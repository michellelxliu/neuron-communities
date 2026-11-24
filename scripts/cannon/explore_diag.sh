#!/bin/bash
# CPU job (test): a/an neuron diagnostic and prefix-aware summaries of the exploratory slices (no GPU).
set -euo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path"); X=$WORK/explore_v1
OUT=outputs/explore_diag_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
for split in discovery heldout; do
  $PY scripts/gates/explore_summary.py --harvest "$X/harvest_$split" --prompts "$X/$split" --out "$OUT/summary_$split.json" --model "$SNAP" > /dev/null
done
$PY scripts/gates/aan_probe.py --harvest "$X/harvest_discovery" --prompts "$X/discovery" --out "$OUT/aan_probe.json"
cp .code_commit "$OUT/" 2>/dev/null || true
echo DIAG_OK
