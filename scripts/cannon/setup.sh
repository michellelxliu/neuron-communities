#!/bin/bash
# CPU job (test partition): venv, pinned model snapshot, the corpus probe sample, CPU test suite.
set -euo pipefail
source scripts/cannon/env.sh
UV=$HOME/.local/bin/uv
echo "== venv"
[ -x "$PY" ] || $UV venv --python 3.12 "$WORK/venv"
$UV pip install -q --python "$PY" --index-url https://download.pytorch.org/whl/cu132 torch==2.15.0
$UV pip install -q --python "$PY" transformers==5.24.0 safetensors datasets numpy scipy python-igraph leidenalg pyarrow pandas pytest huggingface_hub
$PY -c "import torch, transformers, pyarrow; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'transformers', transformers.__version__, 'pyarrow', pyarrow.__version__)"

echo "== model snapshot $MODEL_REPO@$MODEL_REV"
$PY - <<PYEOF
from huggingface_hub import snapshot_download
p = snapshot_download("$MODEL_REPO", revision="$MODEL_REV", allow_patterns=["*.json", "*.safetensors"])
open("$WORK/model_path", "w").write(p)
print(p)
PYEOF
SNAP=$(cat "$WORK/model_path"); du -sh "$SNAP"/ -L

echo "== corpus probe sample (Goodfire sources and settings, small)"
P=$WORK/probe
nc() { $PY -m neuron_communities "$@"; }
[ -f "$P/tokens/fineweb/index.sqlite" ] || nc tokens --hf-dataset HuggingFaceFW/fineweb-edu --hf-config sample-10BT --text-field text --n-docs 200 --seed 2 --model "$SNAP" --out "$P/tokens/fineweb"
[ -f "$P/tokens/wiki/index.sqlite" ] || nc tokens --hf-dataset wikimedia/wikipedia --hf-config 20231101.en --text-field text --n-docs 100 --seed 2 --model "$SNAP" --out "$P/tokens/wiki"
[ -f "$P/tokens/finemath/index.sqlite" ] || nc tokens --hf-dataset HuggingFaceTB/finemath --hf-config finemath-4plus --text-field text --n-docs 100 --seed 2 --model "$SNAP" --out "$P/tokens/finemath"
for c in fineweb:192 wiki:96 finemath:96; do
  name=${c%%:*}; n=${c##*:}
  [ -f "$P/man/$name/manifest.npz" ] || nc sample --dataset "$P/tokens/$name" --out "$P/man/$name" --n "$n" --seed 2
  [ -f "$P/man/$name/dedup_kept.npz" ] || nc dedup --manifest-dir "$P/man/$name"
done
[ -f "$P/base/manifest.npz" ] || nc combine --source "$P/man/fineweb:fineweb" --source "$P/man/wiki:wiki" --source "$P/man/finemath:finemath" --out "$P/base" --tokens-out "$P/tokens/combined"

echo "== CPU tests"
$PY -m pytest -q -p no:warnings tests
echo "SETUP_OK"
