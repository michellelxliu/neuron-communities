#!/bin/bash
# GPU job (gpu_test, 1 MIG 3g.20gb slice): the arithmetic slice. Build the discovery / held-out prompt worlds (Table 7
# templates, disjoint templates and value pairs), harvest both with attribution-wide (top 1,000 per layer, the corpus
# flags), summarise accuracy and layer-18 recovery per task, template and period cluster, upload to Blob.
# Gate 1 measured ~375k arithmetic prompts/h per slice; ~27k prompts here. Resumable (parts skipped when present).
set -uo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path")
A=$WORK/arith_v1
OUT=outputs/arith_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
nvidia-smi -L | tee "$OUT/gpus.txt"
[ -f "$A/slice_summary.json" ] || $PY scripts/slices/build_arithmetic.py --model "$SNAP" --out "$A" || { echo ARITH=build_failed; exit 1; }
cp "$A"/slice_summary.json "$A"/ground_truth.json "$A"/templates.json "$OUT/"
ok=1
for split in discovery heldout; do
  t0=$(date +%s)
  $PY -m neuron_communities attribution-wide --manifest-dir "$A/$split" --out-dir "$A/harvest_$split" --shard 0 --shards 1 \
      --part-blocks 4 --kv-budget-gb 1.5 --forward-token-budget 6144 --crosscheck-max-tokens 512 --split "$split" \
      --model "$SNAP" --device cuda > "$OUT/harvest_$split.log" 2>&1 || { echo "harvest $split FAILED"; tail -20 "$OUT/harvest_$split.log"; ok=0; continue; }
  echo "harvest $split: $(( $(date +%s) - t0 )) s"
  cp "$A/harvest_$split"/run_meta_shard*.json "$OUT/run_meta_$split.json"
  $PY scripts/gates/arith_summary.py --harvest "$A/harvest_$split" --prompts "$A/$split" --truth "$A/ground_truth.json" \
      --out "$OUT/summary_$split.json" && echo "summary $split ok"
done
cp .code_commit "$OUT/" 2>/dev/null || true
[ $ok = 1 ] || { echo ARITH=failed; exit 1; }
$PY scripts/cannon/blob_upload.py --src "$A" --dest stat236/arith_v1 --index "$OUT/blob_index_arith_v1.json" --workers 8
echo ARITH=ok
