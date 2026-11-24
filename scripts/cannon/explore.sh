#!/bin/bash
# GPU job (gpu_test, 1 MIG slice): the exploratory slices (agreement, delimiters, copying). Build both prompt worlds,
# harvest them with attribution-wide (top 1,000 per layer, the corpus flags), summarise, upload to Blob.
set -uo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path")
X=$WORK/explore_v1
OUT=outputs/explore_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
nvidia-smi -L | tee "$OUT/gpus.txt"
[ -f "$X/slice_summary.json" ] || $PY scripts/slices/build_exploratory.py --model "$SNAP" --out "$X" || { echo EXPLORE=build_failed; exit 1; }
cp "$X/slice_summary.json" "$OUT/"
ok=1
for split in discovery heldout; do
  t0=$(date +%s)
  $PY -m neuron_communities attribution-wide --manifest-dir "$X/$split" --out-dir "$X/harvest_$split" --shard 0 --shards 1 \
      --part-blocks 4 --kv-budget-gb 1.5 --forward-token-budget 6144 --crosscheck-max-tokens 512 --split "$split" \
      --model "$SNAP" --device cuda > "$OUT/harvest_$split.log" 2>&1 || { echo "harvest $split FAILED"; tail -20 "$OUT/harvest_$split.log"; ok=0; continue; }
  echo "harvest $split: $(( $(date +%s) - t0 )) s"
  cp "$X/harvest_$split"/run_meta_shard*.json "$OUT/run_meta_$split.json"
  $PY scripts/gates/explore_summary.py --harvest "$X/harvest_$split" --prompts "$X/$split" --out "$OUT/summary_$split.json" \
      && echo "summary $split ok"
done
cp .code_commit "$OUT/" 2>/dev/null || true
[ $ok = 1 ] || { echo EXPLORE=failed; exit 1; }
$PY scripts/cannon/blob_upload.py --src "$X" --dest stat236/explore_v1 --index "$OUT/blob_index_explore_v1.json" --workers 8
echo EXPLORE=ok
