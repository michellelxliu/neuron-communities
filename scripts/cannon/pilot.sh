#!/bin/bash
# GPU job (gpu_test, N MIG 3g.20gb slices in one job): the 100k pilot harvest, data parallel (one shard per slice), then
# gate 3 (truncation curve) and gate 4 (throughput) summaries and the Blob upload of the harvest.
# Same flags as the gate-1 probe (job 62782891), which measured 7,476 corpus prompts/h per slice at 17.6 GiB peak.
# Resumable: parts are atomic and skipped when present, so a resubmission continues where a job stopped.
set -uo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path")
M=$WORK/world_v1/pilot; H=$WORK/world_v1/pilot_harvest
OUT=outputs/pilot_${SLURM_JOB_ID:-local}; mkdir -p "$OUT" "$H/logs"
nvidia-smi -L | tee "$OUT/gpus.txt"
mapfile -t MIG < <(nvidia-smi -L | sed -n 's/.*(UUID: \(MIG-[^)]*\)).*/\1/p')
[ ${#MIG[@]} -gt 0 ] || mapfile -t MIG < <(nvidia-smi -L | sed -n 's/.*(UUID: \(GPU-[^)]*\)).*/\1/p')
N=${#MIG[@]}; echo "devices: $N"
t0=$(date +%s)
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${MIG[$i]} $PY -m neuron_communities attribution-wide --manifest-dir "$M" --out-dir "$H" \
      --shard "$i" --shards "$N" --part-blocks 4 --kv-budget-gb 1.5 --forward-token-budget 6144 \
      --crosscheck-max-tokens 512 --split train --model "$SNAP" --device cuda > "$H/logs/shard_$i.log" 2>&1 &
  pids+=($!)
done
fail=0
for i in "${!pids[@]}"; do wait "${pids[$i]}" || { echo "shard $i FAILED"; tail -20 "$H/logs/shard_$i.log"; fail=1; }; done
echo "harvest wall time: $(( $(date +%s) - t0 )) s on $N devices"
cp "$H"/run_meta_shard*.json "$H"/logs/*.log "$OUT/" 2>/dev/null; cp .code_commit "$OUT/" 2>/dev/null
[ $fail = 0 ] || { echo "PILOT=failed"; exit 1; }
$PY scripts/gates/curve_summary.py --harvest "$H" --out "$OUT/gate3_truncation.json" > "$OUT/gate3_truncation.txt" && echo GATE3_SUMMARY=ok
$PY scripts/gates/probe_summary.py --harvest "$H" --target-prompts 2100000 --slices 16 --out "$OUT/gate4_throughput.json" \
    > "$OUT/gate4_throughput.txt" && echo GATE4_SUMMARY=ok
cat "$OUT/gate4_throughput.txt" | tail -20
$PY scripts/cannon/blob_upload.py --src "$H" --dest stat236/world_v1/pilot_harvest --index "$OUT/blob_index_pilot_harvest.json" --workers 8
echo "PILOT=ok"
