#!/bin/bash
# GPU job (gpu_test, 1 MIG 3g.20gb slice): smoke test of the full-run storage width, top 5,000 neurons per layer.
# Re-harvests the first 6 parts (1,536 prompts) of pilot shard 0 with the pilot's flags except --k 5000, then
#  - measures throughput (prompts/h per slice) and bytes per stored entry, to size the full run;
#  - checks the top-1,000 prefix of each layer against the pilot harvest (job 63410848) on the same prompts.
set -uo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path")
M=$WORK/world_v1/pilot; REF=$WORK/world_v1/pilot_harvest; H=$WORK/world_v1/smoke_k5000_${SLURM_JOB_ID:-local}
OUT=outputs/smoke_k5000_${SLURM_JOB_ID:-local}; mkdir -p "$OUT" "$H/logs"
nvidia-smi -L | tee "$OUT/gpus.txt"
t0=$(date +%s)
$PY -m neuron_communities attribution-wide --manifest-dir "$M" --out-dir "$H" --k 5000 \
    --shard 0 --shards 4 --max-parts 6 --part-blocks 4 --kv-budget-gb 1.5 --forward-token-budget 6144 \
    --crosscheck-max-tokens 512 --split train --model "$SNAP" --device cuda > "$H/logs/shard_0.log" 2>&1
rc=$?
echo "harvest wall time: $(( $(date +%s) - t0 )) s, rc=$rc"
cp "$H"/run_meta_shard*.json "$H/logs/shard_0.log" "$OUT/" 2>/dev/null; cp .code_commit "$OUT/" 2>/dev/null
[ $rc = 0 ] || { tail -30 "$H/logs/shard_0.log"; echo "SMOKE=failed"; exit 1; }
$PY - "$H" "$REF" "$OUT/smoke_k5000.json" <<'EOF'
import json, sys, numpy as np, pyarrow.parquet as pq
from pathlib import Path
new, ref, out = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
L, K, K0 = 32, 5000, 1000
res = {"parts": [], "k": K}
idx_eq = rows = 0; s_err = []; byts = n_prompts = 0; tok = 0
for f in sorted((new / "parts").glob("part_*.parquet")):
    a = pq.read_table(f).to_pandas(); b = pq.read_table(ref / "parts" / f.name).to_pandas()
    assert (a.manifest_index.values == b.manifest_index.values).all(), f.name
    byts += f.stat().st_size; n_prompts += len(a); tok += sum(len(t) for t in a.token_ids)
    for ia, sa, ib, sb in zip(a.idx, a.s, b.idx, b.s):
        ia = np.asarray(ia).reshape(L, K)[:, :K0]; sa = np.asarray(sa).reshape(L, K)[:, :K0]
        ib = np.asarray(ib).reshape(L, K0); sb = np.asarray(sb).reshape(L, K0)
        rows += 1; idx_eq += int((ia == ib).all())
        s_err.append(float(np.max(np.abs(sa - sb)) / max(np.max(np.abs(sb)), 1e-30)))
res.update(prompts=n_prompts, tokens=tok, bytes=byts, bytes_per_prompt=byts / n_prompts,
           bytes_per_entry=(byts - 0) / (n_prompts * L * K),
           top1000_identical_rows=idx_eq, rows=rows, s_max_rel_err=max(s_err), s_median_rel_err=float(np.median(s_err)))
ref_bytes = sum((ref / "parts" / f.name).stat().st_size for f in sorted((new / "parts").glob("part_*.parquet")))
res["ref_bytes_per_prompt_k1000"] = ref_bytes / n_prompts
json.dump(res, open(out, "w"), indent=1); print(json.dumps(res, indent=1))
EOF
$PY scripts/gates/probe_summary.py --harvest "$H" --target-prompts 2000000 --slices 8 --out "$OUT/throughput.json" \
    > "$OUT/throughput.txt"; tail -20 "$OUT/throughput.txt"
echo "SMOKE=ok"
