#!/bin/bash
# GPU job (gpu_test, one MIG 3g.20gb slice): gate 1, then the memory/throughput probe on corpus prompts.
set -uo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path")
OUT=outputs/gate1_probe_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
nvidia-smi --query-gpu=name,memory.total --format=csv | tee "$OUT/gpu.txt"
echo "== gate 1"
$PY scripts/gates/gate1_arith.py --model "$SNAP" --out "$WORK/gate1" --device cuda && G1=ok || G1=failed
cp "$WORK/gate1/gate1.json" "$WORK/gate1/gate1.md" "$OUT/" 2>/dev/null; cp "$WORK"/gate1/harvest/run_meta_shard*.json "$OUT/" 2>/dev/null
echo "== probe: corpus prompts, reduced batch budgets, crosscheck capped at 512 tokens"
rm -rf "$WORK/probe/harvest"
$PY -m neuron_communities attribution-wide --manifest-dir "$WORK/probe/base" --out-dir "$WORK/probe/harvest" --shard 0 --shards 1 \
    --part-blocks 1 --kv-budget-gb 1.5 --forward-token-budget 6144 --crosscheck-max-tokens 512 --split probe \
    --model "$SNAP" --device cuda && PR=ok || PR=failed
$PY scripts/gates/probe_summary.py --harvest "$WORK/probe/harvest" --target-prompts 2100000 --slices 16 --out "$OUT/probe_summary.json"
echo "GATE1=$G1 PROBE=$PR"
