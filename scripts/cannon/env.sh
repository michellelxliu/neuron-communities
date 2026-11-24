# Shared environment of the Cannon jobs (sourced). Work dir on netscratch (purged; not a record: results go to Blob).
export WORK=${WORK:-/n/netscratch/hoffman_lab/Lab/michelleliu/stat236}
export HF_HOME=$WORK/hf
export HF_TOKEN_PATH=$HOME/.cache/huggingface/token     # the token stays in $HOME; HF_HOME moves the cache only
export UV_CACHE_DIR=$WORK/uv-cache
export PY=$WORK/venv/bin/python
export MODEL_REPO=${MODEL_REPO:-meta-llama/Llama-3.1-8B}   # or NousResearch/Meta-Llama-3.1-8B @ 1f47e50cdbe801ad8a5174156ec3a0655108fb9f (byte-identical weights, ungated)
export MODEL_REV=${MODEL_REV:-d04e592bb4f6aa9cfee91e2e20afa771667e1d4b}
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$WORK"
