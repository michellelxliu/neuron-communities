# Full harvest (`full_v1`): running it together

The full harvest attributes every deduplicated corpus prompt of `world_v1`, keeping the top 5,000 neurons per layer:

| Split | Prompts | Parts (256 prompts) | Chunks (32 parts) |
|---|---|---|---|
| `train` | 979,047 | 3,825 | 120 |
| `heldout` | 977,991 | 3,821 | 120 |

Several people can run it at the same time, on any machines. The work is shared through the `stat236` Blob container,
so nobody needs to agree on a split in advance:

- **No overlap.** A worker claims the next free chunk by creating `full_v1/claims/<split>_<chunk>.json` with an
  atomic create-only write. If two workers try the same chunk, exactly one succeeds and the other moves on.
- **No overwrites.** Every upload is create-only, with its size and MD5 read back. Nothing in `scripts/full/` replaces
  a blob that exists. Use only these scripts to write under `full_v1/`; other tools (azcopy, the portal) would
  overwrite.
- **Same code, same run.** `full_v1/config.json` pins the code (commit, and a hash of `neuron_communities/*.py`
  from `scripts/full/code_hash.py`), the harvest flags and the model. A worker refuses to start if its code differs.
- **Crashes.** A chunk is marked done only after all its parts are verified in Blob. If a job dies, its claims stay.
  Rerun with the same `OWNER` and your claims from dead jobs are taken back. If you stop for good, release them
  (below) and someone else picks them up.

## Blob layout (`full_v1/`)

| Path | What |
|---|---|
| `config.json` | the run config (written once by Michelle; the run starts when it exists) |
| `inputs.json` | the input files (5.8 GB) and token-chunk links, read by `fetch_inputs.py` |
| `claims/`, `done/` | the work queue: one small JSON per chunk |
| `<split>/parts/part_XXXXXX.parquet` | the harvest, same format as `world_v1/pilot_harvest/` (see `DATA.md`), `--k 5000` |
| `<split>/run_meta/`, `logs/<owner>/` | each chunk's run metadata (code commit, library versions, GPU) and log |

## Setup (once)

1. **Code:** `git clone -b wide-harvest https://github.com/michellelxliu/neuron-communities.git`.
2. **Python 3.12 env** with the versions of the Cannon runs: torch 2.15.0 (CUDA 13.2 wheels), transformers 5.24.0,
   plus safetensors, numpy, scipy, pyarrow, pandas, huggingface_hub. `scripts/cannon/setup.sh` builds it on Cannon.
   Other versions run, but record them; the run metadata stores what you used.
3. **Model:** Llama-3.1-8B, bf16. Without access to the gated Meta repo, use the byte-identical mirror
   `NousResearch/Meta-Llama-3.1-8B` at revision `1f47e50cdbe801ad8a5174156ec3a0655108fb9f` (same sha256 for every
   weight file). On Cannon: `export MODEL_REPO=NousResearch/Meta-Llama-3.1-8B MODEL_REV=1f47e50cdbe801ad8a5174156ec3a0655108fb9f`
   before `setup.sh`.
4. **Blob link:** save the collaborator link from Michelle in a private file, never in the repo:
   ```bash
   mkdir -p ~/.config/stat236-blob && chmod 700 ~/.config/stat236-blob
   printf 'STAT236_COLLAB_SAS=%s\n' '<the part after ? in the link>' > ~/.config/stat236-blob/collab.env
   chmod 600 ~/.config/stat236-blob/collab.env
   export STAT236_SAS_FILE=~/.config/stat236-blob/collab.env:STAT236_COLLAB_SAS
   ```
5. **Inputs** (5.8 GB, a minute or two inside Azure, longer elsewhere):
   `python scripts/full/fetch_inputs.py --dest $WORK` (prints `FETCH_OK`).

## Run

Pick a short, unique `OWNER` name (e.g. your first name) and keep it for the whole run.

**On Cannon (gpu_test, up to 8 MIG slices per person):**

```bash
FAS_EXEMPT_ONLY=1 FAS_NODEPS=1 fas gpu env OWNER=<name> WORK=$WORK STAT236_SAS_FILE=$STAT236_SAS_FILE bash scripts/cannon/full_harvest.sh
```

or the equivalent `sbatch` with `--gres=gpu:4` on `gpu_test`, 12 h. One worker runs per slice, and each takes about
1 h per chunk (about 8,000 prompts per hour per slice). Resubmit the same command when the job ends; it continues.

**Anywhere else, one worker per GPU:**

```bash
OWNER=<name> JOB_ID=<host>-gpu0 WORK=$WORK MODEL=<snapshot dir> CUDA_VISIBLE_DEVICES=0 \
  MEM_FLAGS="--kv-budget-gb 8 --forward-token-budget 49152" bash scripts/full/worker.sh
```

`MEM_FLAGS` only change batching (the default fits a 20 GB slice), not the results. Set `DEADLINE` (epoch seconds) if
the machine has a time limit; then no chunk is started that would not finish.

## Check, release

```bash
python scripts/full/blobq.py status -v          # done / in progress / free, per owner
python scripts/full/blobq.py release --split train --chunk 17 --owner <name>   # give back a claim you will not finish
```

When a worker dies, find its claims with `status -v`. Release them, or rerun with the same `OWNER`: the new job
takes back claims of yours whose job id is not running. On Cannon `full_harvest.sh` passes your running jobs
automatically; elsewhere pass `LIVE_JOBS` (a comma list of your running `JOB_ID`s).
