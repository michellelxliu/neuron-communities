#!/bin/bash
# CPU job (test): export pruned harvests (scripts/export/export_shared.py) and upload them to the private dataset
# stat236/llama31-8b-neuron-attributions. Usage: hf_publish.sh <config>... with config in arithmetic, exploratory,
# corpus_pilot. The token is read from ~/.config/stat236-hf/token (never printed).
set -euo pipefail
source scripts/cannon/env.sh
E=$WORK/hf_export; mkdir -p "$E/data" "$E/aux"
OUT=outputs/hf_publish_${SLURM_JOB_ID:-local}; mkdir -p "$OUT"
for cfg in "$@"; do
  case $cfg in
    arithmetic)  for s in discovery heldout; do $PY scripts/export/export_shared.py --harvest "$WORK/arith_v1/harvest_$s" --prompts "$WORK/arith_v1/$s" --out "$E/data/arithmetic" --split $s; done
                 mkdir -p "$E/aux/arithmetic"; cp "$WORK"/arith_v1/{ground_truth.json,templates.json,slice_summary.json} "$E/aux/arithmetic/" ;;
    exploratory) for s in discovery heldout; do $PY scripts/export/export_shared.py --harvest "$WORK/explore_v1/harvest_$s" --prompts "$WORK/explore_v1/$s" --out "$E/data/exploratory" --split $s; done
                 mkdir -p "$E/aux/exploratory"; cp "$WORK/explore_v1/slice_summary.json" "$E/aux/exploratory/" ;;
    corpus_pilot) $PY scripts/export/export_shared.py --harvest "$WORK/world_v1/pilot_harvest" --prompts "$WORK/world_v1/pilot" --out "$E/data/corpus_pilot" --split train
                 mkdir -p "$E/aux/corpus_pilot"; cp "$WORK/world_v1/pilot/meta.json" "$E/aux/corpus_pilot/pilot_meta.json"
                 cp "$WORK"/world_v1/pilot_harvest/run_meta_shard*.json "$E/aux/corpus_pilot/" ;;
    *) echo "unknown config $cfg"; exit 1 ;;
  esac
done
$PY scripts/export/hf_card.py "$E"
cp "$E"/data/*/*_export.json "$OUT/" 2>/dev/null || true
HF_TOKEN=$(cat ~/.config/stat236-hf/token) $PY - "$E" <<'PYEOF'
import os, sys
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = "stat236/llama31-8b-neuron-attributions"
info = api.upload_folder(repo_id=repo, repo_type="dataset", folder_path=sys.argv[1],
                         allow_patterns=["README.md", "data/**", "aux/**"], ignore_patterns=["**/*_export.json"],
                         commit_message="Add/update pruned harvests: " + ", ".join(sorted(os.listdir(os.path.join(sys.argv[1], "data")))))
files = api.list_repo_files(repo, repo_type="dataset")
print("HF_UPLOAD_OK", info.oid if hasattr(info, "oid") else info, "files:", len(files))
PYEOF
echo HF_PUBLISH_DONE
