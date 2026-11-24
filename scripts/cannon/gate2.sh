#!/bin/bash
# CPU job (test partition): gate 2. The pilot harvest's top-5 prefix in the upstream block layout (wide-blocks --k 5),
# then upstream world -> census (465 tiles, layers 0-29) -> leiden (CPM grid) -> merge -> hubs, unchanged and with the
# paper's settings, then the a/an community check. Every stage skips outputs that exist, so a resubmission continues.
#   TILE_JOBS tiles run at once, each with CENSUS_WORKERS null-replicate processes.
set -uo pipefail
source scripts/cannon/env.sh
TILE_JOBS=${TILE_JOBS:-8}; CENSUS_WORKERS=${CENSUS_WORKERS:-12}
M=$WORK/world_v1/pilot; H=$WORK/world_v1/pilot_harvest; G=$WORK/world_v1/pilot_gate2
H5=$G/harvest_top5; W=$G/world
OUT=outputs/gate2_${SLURM_JOB_ID:-local}; mkdir -p "$OUT" "$G/logs"
nc() { echo "+ neuron_communities $*"; $PY -m neuron_communities "$@"; }
set -e
[ -d "$H5/blocks" ] || nc wide-blocks --harvest-dir "$H" --out-dir "$H5" --k 5
[ -d "$W/merged" ] || nc world --harvest-dir "$H5" --manifest-dir "$M" --out-dir "$W"
NT=$($PY -m neuron_communities census --world "$W" --n-tiles | tail -1)
echo "== census: $NT tiles, $TILE_JOBS at once x $CENSUS_WORKERS workers"
t0=$(date +%s)
# the diagonal tiles first (largest), then the rest; xargs keeps TILE_JOBS running
seq 0 $((NT - 1)) | xargs -P "$TILE_JOBS" -I{} sh -c \
  "$PY -m neuron_communities census --world '$W' --tile {} --workers $CENSUS_WORKERS > '$G/logs/census_{}.log' 2>&1 || echo 'tile {} FAILED'"
echo "census wall time: $(( $(date +%s) - t0 )) s"
n_done=$(ls "$W"/census/pairs_L*_L*.npz | wc -l); echo "tiles done: $n_done / $NT"; [ "$n_done" = "$NT" ]
CD=$W/cand_cpm
NR=$($PY -m neuron_communities leiden --census-dir "$W/census" --out-dir "$CD" --n-runs | tail -1)
seq 0 $((NR - 1)) | xargs -P 8 -I{} sh -c \
  "[ -f '$CD/runs/'\$(printf 'run_%03d.json.gz' {}) ] || $PY -m neuron_communities leiden --census-dir '$W/census' --out-dir '$CD' --run {} > '$G/logs/leiden_{}.log' 2>&1 || echo 'leiden run {} FAILED'"
[ -f "$CD/candidates.json.gz" ] || nc leiden --census-dir "$W/census" --out-dir "$CD" --merge-runs
[ -f "$W/candidates_with_hubs/candidates.json.gz" ] || nc merge --inputs "$CD/candidates.json.gz" --out "$W/candidates_with_hubs"
[ -f "$W/candidates/candidates.json.gz" ] || nc hubs --out "$W/candidates" --runs "$CD/runs"
$PY scripts/gates/gate2_aan.py --world "$W" --out "$OUT/gate2_aan.json" | tee "$OUT/gate2_aan.txt"
cp "$W"/meta.json "$CD"/candidates_stats.json "$OUT/" 2>/dev/null || true; cp .code_commit "$OUT/" 2>/dev/null || true
$PY scripts/cannon/blob_upload.py --src "$G" --dest stat236/world_v1/pilot_gate2 --index "$OUT/blob_index_pilot_gate2.json" \
    --skip-glob 'harvest_top5/*' --workers 8
echo "GATE2_DONE"
