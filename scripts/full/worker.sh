#!/bin/bash
# One full-harvest worker on one GPU (or MIG slice): claim the next free chunk in Blob, harvest it, upload its parts
# (create-only, MD5 read back), mark it done, repeat until no chunk is left or the deadline is near.
# Several workers (several people, machines, slices) can run at once; the Blob claims keep their chunks disjoint.
#
# Required env:  OWNER (short name, e.g. michelle), JOB_ID (unique per worker, e.g. <slurm id>.<device>),
#                WORK (dir with world_v1/ from fetch_inputs.py; parts are staged in $WORK/full_v1),
#                MODEL (local Llama-3.1-8B snapshot dir), STAT236_SAS_FILE (FILE:VAR holding the container SAS)
# Optional env:  PY (python, default python3), DEADLINE (epoch s: claim no new chunk after DEADLINE - CHUNK_SECONDS),
#                CHUNK_SECONDS (default 5400), LIVE_JOBS (comma list of this OWNER's running JOB_IDs: claims of yours
#                under other ids are from dead jobs and are taken back), MEM_FLAGS (memory-only harvest flags; default
#                fits a 20 GB A100 MIG slice; outputs do not depend on them), RUN (default full_v1)
set -uo pipefail
: "${OWNER:?}" "${JOB_ID:?}" "${WORK:?}" "${MODEL:?}" "${STAT236_SAS_FILE:?}"
PY=${PY:-python3}; RUN=${RUN:-full_v1}; CHUNK_SECONDS=${CHUNK_SECONDS:-5400}; DEADLINE=${DEADLINE:-0}
MEM_FLAGS=${MEM_FLAGS:---kv-budget-gb 1.5 --forward-token-budget 6144}
export STAT236_SAS_FILE
Q="$PY scripts/full/blobq.py --run $RUN"
cfg=$($Q get-config) || { echo "worker: no run config in Blob"; exit 1; }
cfgv() { echo "$cfg" | $PY -c "import json,sys; d=json.load(sys.stdin); print(eval(sys.argv[1], {}, {'c': d}))" "$1"; }
FLAGS=$(cfgv 'c["harvest_flags"]'); WANT=$(cfgv 'c["code_hash"]')

# the harvest code (neuron_communities/*.py) must be the run's; works in git checkouts and synced copies alike
HAVE=$($PY scripts/full/code_hash.py)
[ "$HAVE" = "$WANT" ] || { echo "worker: neuron_communities hash $HAVE differs from the run's $WANT (commit $(cfgv 'c["commit"]'))"; exit 1; }

LOGD=$WORK/$RUN/logs; mkdir -p "$LOGD"
echo "worker $OWNER/$JOB_ID on $(hostname) device ${CUDA_VISIBLE_DEVICES:-?} start $(date -u +%FT%TZ)"
n=0
while true; do
  if [ "$DEADLINE" -gt 0 ] && [ $(( $(date +%s) + CHUNK_SECONDS )) -gt "$DEADLINE" ]; then
    echo "worker: deadline near, stopping after $n chunks"; break
  fi
  spec=$($Q claim --owner "$OWNER" --job "$JOB_ID" --live-jobs "${LIVE_JOBS:-}"); rc=$?
  [ $rc = 3 ] && { echo "worker: no free chunks left, $n done"; break; }
  [ $rc = 0 ] || { echo "worker: claim failed rc=$rc"; exit 1; }
  read -r split chunk lo hi <<< "$spec"
  H=$WORK/$RUN/$split; log=$LOGD/${split}_$(printf %05d "$chunk").log
  echo "$(date -u +%TZ) chunk $split $chunk parts $lo..$((hi - 1))"
  t0=$(date +%s)
  # --shard = chunk id only names this chunk's run_meta file; --part-range selects the parts
  $PY -m neuron_communities attribution-wide --manifest-dir "$WORK/world_v1/$split/full" --tokens "$WORK/world_v1/$split/tokens" \
      --out-dir "$H" --split "$split" --shard "$chunk" --shards 1 --part-range "$lo:$hi" $FLAGS $MEM_FLAGS \
      --model "$MODEL" --device cuda >> "$log" 2>&1
  [ $? = 0 ] || { echo "worker: harvest failed on $split $chunk (claim kept; a rerun takes it back)"; tail -20 "$log"; exit 1; }
  t1=$(date +%s); up=0; bytes=0
  for p in $(seq "$lo" $((hi - 1))); do
    f=$H/parts/part_$(printf %06d "$p").parquet
    [ -f "$f" ] || { echo "worker: missing $f"; exit 1; }
    r=$($Q put "$f" "$RUN/$split/parts/$(basename "$f")") || { echo "worker: upload failed $f"; exit 1; }
    bytes=$(( bytes + $(stat -c %s "$f") )); up=$((up + 1))
  done
  meta=$H/run_meta_shard$(printf %04d "$chunk").json
  [ -f "$meta" ] && $Q put "$meta" "$RUN/$split/run_meta/chunk_$(printf %05d "$chunk")_${OWNER}_${JOB_ID}.json" > /dev/null
  $Q put "$log" "$RUN/logs/$OWNER/${split}_$(printf %05d "$chunk")_${JOB_ID}.log" > /dev/null
  $Q done --split "$split" --chunk "$chunk" --owner "$OWNER" \
      --info "{\"job\": \"$JOB_ID\", \"host\": \"$(hostname)\", \"parts\": $up, \"bytes\": $bytes, \"harvest_s\": $((t1 - t0)), \"upload_s\": $(( $(date +%s) - t1 ))}"
  for p in $(seq "$lo" $((hi - 1))); do rm -f "$H/parts/part_$(printf %06d "$p").parquet"; done   # verified in Blob
  n=$((n + 1))
  echo "$(date -u +%TZ) chunk $split $chunk done: harvest $((t1 - t0)) s, upload $(( $(date +%s) - t1 )) s, $((bytes / 1000000)) MB"
done
echo "WORKER_END $OWNER/$JOB_ID chunks=$n"
