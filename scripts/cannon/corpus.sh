#!/bin/bash
# CPU job (test partition): the Build A corpus with upstream's sources and settings, at 1M training + 1M held-out prompts.
#   tokens (paper doc counts, now with docs.parquet source doc ids) -> sample + dedup per corpus and split -> heldout-dedup
#   -> combine -> short-context dedup -> prompt-meta -> the 100k pilot subset of the training world (50/25/25k)
#   then the tokens datasets and manifests go to Blob stat236/world_v1/ (verified; index in outputs/).
# Prompt counts per corpus: half of each upstream default (5M / 2.5M / 2.5M are the paper's 10M) divided by 5, i.e. the
# paper's 50/25/25 mix at 1M per split. Seeds: train 2, held-out 3 (upstream defaults), pilot shuffle 0.
set -euo pipefail
source scripts/cannon/env.sh
SNAP=$(cat "$WORK/model_path")
W=$WORK/world_v1
OUT=outputs/corpus_${SLURM_JOB_ID:-local}; mkdir -p "$OUT" "$W/logs"
nc() { echo "+ neuron_communities $*"; $PY -m neuron_communities "$@"; }
$PY -c "import azure.storage.blob" 2>/dev/null || $HOME/.local/bin/uv pip install -q --python "$PY" azure-storage-blob

declare -A HF=([fineweb]="HuggingFaceFW/fineweb-edu sample-10BT" [wiki]="wikimedia/wikipedia 20231101.en" [finemath]="HuggingFaceTB/finemath finemath-4plus")
declare -A DOCS=([fineweb]=700000 [wiki]=400000 [finemath]=400000)
declare -A NPROMPTS=([fineweb]=500000 [wiki]=250000 [finemath]=250000)
declare -A PILOT=([fineweb]=50000 [wiki]=25000 [finemath]=25000)
C="fineweb wiki finemath"

echo "== tokens (three corpora in parallel)"
pids=()
for c in $C; do
  if [ ! -f "$W/tokens/$c/docs.parquet" ]; then
    set -- ${HF[$c]}
    nc tokens --hf-dataset "$1" --hf-config "$2" --text-field text --n-docs "${DOCS[$c]}" --seed 2 --model "$SNAP" \
       --out "$W/tokens/$c" > "$W/logs/tokens_$c.log" 2>&1 &
    pids+=($!)
  fi
done
for p in "${pids[@]}"; do wait "$p"; done
for c in $C; do tail -1 "$W/logs/tokens_$c.log"; test -f "$W/tokens/$c/docs.parquet"; done

echo "== sample + dedup"
for split in train heldout; do
  seed=2; [ $split = heldout ] && seed=3
  for c in $C; do
    M=$W/manifests/${c}_$split
    [ -f "$M/manifest.npz" ] || nc sample --dataset "$W/tokens/$c" --out "$M" --n "${NPROMPTS[$c]}" --seed $seed
    [ -f "$M/dedup_kept.npz" ] || nc dedup --manifest-dir "$M"
  done
done
for c in $C; do
  M=$W/manifests/${c}_heldout
  [ -f "$M/cross_dedup.json" ] || nc heldout-dedup --new "$M" --old "$W/manifests/${c}_train"
done

echo "== combine + prompt-meta"
for split in train heldout; do
  B=$W/$split/base
  [ -f "$B/manifest.npz" ] || nc combine --source "$W/manifests/fineweb_$split:fineweb" --source "$W/manifests/wiki_$split:wiki" \
      --source "$W/manifests/finemath_$split:finemath" --out "$B" --tokens-out "$W/$split/tokens"
  [ -f "$B/dedup_drop.npz" ] || nc dedup --manifest-dir "$B" --short-only
  [ -f "$B/prompts.parquet" ] || nc prompt-meta --manifest-dir "$B" --label "$split" --split "$split"
done

echo "== pilot subset (training world)"
[ -f "$W/pilot/manifest.npz" ] || nc subset --parent "$W/train/base" --out "$W/pilot" \
    --per-source "fineweb=${PILOT[fineweb]},wiki=${PILOT[wiki]},finemath=${PILOT[finemath]}" --seed 0

echo "== record"
for c in $C; do cp "$W/tokens/$c/provenance.json" "$OUT/provenance_$c.json"; cp "$W/logs/tokens_$c.log" "$OUT/"; done
for d in manifests/*; do [ -f "$W/$d/cross_dedup.json" ] && cp "$W/$d/cross_dedup.json" "$OUT/cross_dedup_$(basename $d).json"; done
cp "$W/train/base/meta.json" "$OUT/train_meta.json"; cp "$W/heldout/base/meta.json" "$OUT/heldout_meta.json"; cp "$W/pilot/meta.json" "$OUT/pilot_meta.json"
cp .code_commit "$OUT/" 2>/dev/null || true
$PY - "$W" > "$OUT/corpus_summary.json" <<'PYEOF'
import json, sys
from pathlib import Path
import numpy as np
from neuron_communities.data import io
W = Path(sys.argv[1]); s = {}
for c in ("fineweb", "wiki", "finemath"):
    p = json.loads((W / "tokens" / c / "provenance.json").read_text())
    s[f"tokens_{c}"] = {k: p[k] for k in ("n_docs", "total_tokens", "n_docs_with_id")}
for split in ("train", "heldout"):
    B = W / split / "base"; m = io.load_meta(B)
    kept, drop = io.load_kept(B), io.load_drop(B)
    src = np.load(B / "manifest.npz")["source"]; usable = np.delete(kept, drop)
    s[split] = {"n": m["n"], "kept": int(len(kept)), "short_drop": int(len(drop)), "usable": int(len(usable)),
                "usable_per_source": {t: int((src[usable] == v["source_id"]).sum()) for t, v in m["sources"].items()}}
s["pilot"] = io.load_meta(W / "pilot")["counts"]
print(json.dumps(s, indent=1))
PYEOF
cat "$OUT/corpus_summary.json"

echo "== Blob upload (tokens + manifests; the combined tokens dirs are symlinks to the per-corpus chunks and are skipped)"
$PY scripts/cannon/blob_upload.py --src "$W" --dest stat236/world_v1 --index "$OUT/blob_index_world_v1.json" \
    --skip-glob 'train/tokens/*' --skip-glob 'heldout/tokens/*' --workers 8
echo "CORPUS_OK"
