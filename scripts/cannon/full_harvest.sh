#!/bin/bash
# GPU job (gpu_test, N MIG slices): one scripts/full/worker.sh per slice, all pulling chunks from the shared Blob queue
# (full_v1). Resubmit the same command to continue; claims of this owner's dead jobs are taken back.
#   FAS_EXEMPT_ONLY=1 FAS_NODEPS=1 fas gpu env OWNER=michelle bash scripts/cannon/full_harvest.sh
# A collaborator on Cannon sets OWNER, WORK (their copy from fetch_inputs.py), MODEL and STAT236_SAS_FILE the same way.
set -uo pipefail
source scripts/cannon/env.sh
export OWNER=${OWNER:?set OWNER} WORK=${WORK_FULL:-$WORK}
export MODEL=${MODEL:-$(cat "$WORK/model_path")} PY STAT236_SAS_FILE=${STAT236_SAS_FILE:-$HOME/.config/stat236-blob/sas.env:STAT236_SAS}
OUT=outputs/full_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
nvidia-smi -L | tee "$OUT/gpus.txt"
mapfile -t DEV < <(nvidia-smi -L | sed -n 's/.*(UUID: \(MIG-[^)]*\)).*/\1/p')
[ ${#DEV[@]} -gt 0 ] || mapfile -t DEV < <(nvidia-smi -L | sed -n 's/.*(UUID: \(GPU-[^)]*\)).*/\1/p')
N=${#DEV[@]}
# end of this job's time limit; workers claim no chunk that would not finish before it
end=$(squeue -j "${SLURM_JOB_ID}" -h -o %e 2>/dev/null); export DEADLINE=$(( $(date -d "$end" +%s 2>/dev/null || echo 0) - 900 ))
# this owner's running Slurm jobs (workers are named <job>.<device>), so claims of dead jobs can be taken back;
# read again before every claim (scripts/full/worker.sh), so jobs started after this one count as alive
export LIVE_JOBS_CMD='squeue --me -h -o %i | while read -r j; do for i in $(seq 0 15); do printf "%s.%s," "$j" "$i"; done; done'
echo "devices $N, deadline $(date -u -d @$DEADLINE +%FT%TZ), owner $OWNER"
pids=()
for i in $(seq 0 $((N - 1))); do
  CUDA_VISIBLE_DEVICES=${DEV[$i]} JOB_ID=${SLURM_JOB_ID:-local}.$i bash scripts/full/worker.sh > "$OUT/worker_$i.log" 2>&1 &
  pids+=($!); sleep 20   # stagger model loads
done
for i in "${!pids[@]}"; do wait "${pids[$i]}"; echo "worker $i exit $?: $(tail -1 "$OUT/worker_$i.log")"; done
$PY scripts/full/blobq.py --run ${RUN:-full_v1} status -v
echo "FULL_JOB_END"
