#!/bin/bash
# GPU job (gpu_test, one MIG slice): truncation curves on the probe corpus prompts and the gate-1 arithmetic prompts.
set -uo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path")
OUT=outputs/curve_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
for w in probe:$WORK/probe/base gate1:$WORK/gate1/prompts; do
  name=${w%%:*}; man=${w#*:}; H=$WORK/curve/$name
  rm -rf "$H"
  $PY -m neuron_communities attribution-wide --manifest-dir "$man" --out-dir "$H" --shard 0 --shards 1 --part-blocks 1 \
      --kv-budget-gb 1.5 --forward-token-budget 6144 --crosscheck-max-tokens 512 --split "$name" --model "$SNAP" --device cuda \
      && $PY scripts/gates/curve_summary.py --harvest "$H" --out "$OUT/curve_$name.json" > /dev/null && echo "CURVE_$name=ok" || echo "CURVE_$name=failed"
done
