# Stat 236 data: Llama-3.1-8B neuron attributions

Data for extending Goodfire's *Large-scale Neuron Community Discovery* (Shilov et al., arXiv:2611.23112; code
github.com/goodfire-ai/neuron-communities, forked at github.com/michellelxliu/neuron-communities, branch
`wide-harvest`). Every row is one prompt, attributed at its final position in `meta-llama/Llama-3.1-8B` @
`d04e592bb4f6aa9cfee91e2e20afa771667e1d4b` (bf16). The code commit of each run is in its `run_meta_shard*.json`.

## Where it is

| Copy | What | Access |
|---|---|---|
| Azure Blob, container `stat236` (account `infinigenexploration`) | everything, full width (top 1,000 neurons per layer), plus corpora, logs, configs | SAS link from Michelle |
| Hugging Face dataset (private, org) | the harvests, pruned (see below), with prompt metadata joined | org membership |

`catalog.json` at the container root lists every Blob file with its size and MD5. The collaborator link can read
files but cannot list the container, so take paths from the catalog.

## What a row holds

The attributed score of MLP neuron *i* in layer *l* is `s = h · ∂ log p(ŷ | x) / ∂h`, where `h` is the neuron's
SwiGLU activation at the final position and `ŷ` the model's top-1 next token (Goodfire's score is |s|). Every row has:

- identity: `prompt_id`, `split`, `slice`, `task`, `template_id`, `doc_id` (corpus prompts: `<corpus>:<source id>`),
  `row_index`, `position` (the target's offset in its document), `token_ids` (the prompt, BOS first);
- model output: `y_hat`, `p_y_hat`, `top_ids` / `top_logprobs` (top 20 clean next-token log-probs);
- attributions, 32 layers, layer-major, each layer in descending |s|: `idx` (uint16 neuron index), `s` (float32),
  `h` (float16). Full width: 1,000 per layer. Pruned: `n_kept[l]` per layer, = max(50, #neurons with
  |s| ≥ 0.1 × the layer's max |s|);
- truncation statistics over all 14,336 neurons per layer (present in both widths): `s_abs_sum`, `s_sum`,
  `share_at_k` (share of Σ|s| in the top 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000), `k_at_share` (smallest k
  for 50/80/90/95%), `n_above_rel` (neurons with |s| ≥ 0.1 / 0.01 / 0.001 × max);
- slices only: `text`, `answer`, `answer_token`, and for agreement prompts `answer_alt` / `alt_token`.

Global neuron id (Goodfire's convention): `layer * 14336 + idx`.

## Datasets

| Name | Blob prefix | Rows | Notes |
|---|---|---|---|
| corpus world | `world_v1/` | 1.5M docs / 1.37B tokens; 979k train + 978k held-out prompts | Goodfire's sources and dedup (FineWeb-Edu sample-10BT 700k docs, Wikipedia 20231101.en 400k, FineMath-4plus 400k); `tokens/<corpus>/docs.parquet` maps rows to source ids |
| pilot harvest | `world_v1/pilot_harvest/` | 100,000 | 50k / 25k / 25k sample of the training prompts |
| arithmetic slice | `arith_v1/` | 16,575 discovery + 10,732 held-out | Feucht et al. (arXiv:2605.01148) tasks and Table 7 templates; disjoint templates and value pairs between splits; `ground_truth.json` = N_add, per-task sets, period clusters |
| exploratory slices | `explore_v1/` | 11,114 discovery + 11,130 held-out | agreement (a/an, there is/are, subject–verb, fr/es/de), closing delimiters (code, HTML, LaTeX), in-context copying |
| run outputs | `outputs/<run>/` | — | summaries, gate results, logs |

Licences of the sources: FineWeb-Edu and FineMath ODC-By 1.0, Wikipedia CC BY-SA 4.0; model outputs under the
Llama 3.1 Community License. Keep the data within the project until the paper's release decision.

## Loading

From Hugging Face (org members, after `huggingface-cli login`):

```python
from datasets import load_dataset
arith = load_dataset("<org>/<repo>", "arithmetic", split="discovery")
row = arith[0]
L, kept = 18, row["n_kept"]
start = sum(kept[:L]); layer18 = list(zip(row["idx"][start:start + kept[L]], row["s"][start:start + kept[L]]))
```

From Blob with the collaborator link (`SAS` is the part after `?`; it has no list permission, so download by path):

```python
import json, urllib.request
BASE, SAS = "https://infinigenexploration.blob.core.windows.net/stat236", "<sas>"
cat = json.load(urllib.request.urlopen(f"{BASE}/catalog.json?{SAS}"))
for f in cat["files"]:
    if f["path"].startswith("arith_v1/harvest_discovery/"):
        print(f["path"], f["bytes"])   # then: azcopy copy "{BASE}/{path}?{SAS}" <local path>
```

```python
import pyarrow.parquet as pq   # full-width parts: idx/s/h are fixed length 32 * 1000
t = pq.read_table("harvest_discovery/parts/part_000000.parquet")
```

The repo's `neuron_communities.harvest.wide_blocks.load_wide(<harvest dir>)` returns the full-width parts as
(prompts, 32, 1000) arrays, and `wide-blocks --k 5` converts them to Goodfire's top-5 input format.
