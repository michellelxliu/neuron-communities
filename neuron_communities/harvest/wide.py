"""Wide attribution harvest: the top-``k`` (default 1,000) MLP neurons per layer with signed scores and activations.

Same score and the same mechanics as the ``attribution`` stage (``neuron_communities/harvest/attribution.py``, left
unchanged as the baseline): for every manifest target, ``s = h * dL/dh`` with ``h`` the SwiGLU neuron activation at
the target position and ``L = log p(top-1 next token)``; neurons are ranked by ``|s| = |h| * |dL/dh|`` (bit for bit
the upstream ``val_act_ga``). The differences are only in what is kept:

* ``k`` neurons per layer (default 1,000; at most d_mlp) instead of 5, with the **signed** ``s`` and the activation ``h``;
* per prompt: the token ids, the clean top-1 token and probability, the clean top-``--top-logprobs`` log-probs, and
  per layer the totals ``sum |s|`` and ``sum s`` over all neurons (for the truncation check);
* prompt metadata (prompt_id, slice, task, template_id, split, doc_id) joined from an optional
  ``<manifest-dir>/prompts.parquet`` (column ``manifest_index`` plus any of those fields).

The batching (length-sorted sub-batches of a ``--block-size`` block under the K/V and token budgets) and the per
sub-batch crosscheck against a naive full-sequence backward are the upstream ones, so on the same hardware the top-5
prefix of every row equals the upstream stage's output exactly. ``wide-blocks`` writes that prefix in the upstream
block layout, so ``world`` and every later stage run unchanged on it (``tests/test_wide.py`` checks the equality).

Output: ``<out-dir>/parts/part_XXXXXX.parquet``, one per ``--part-blocks`` consecutive blocks (atomic,
skip-if-exists), one row per prompt::

  manifest_index int64, block int32, prompt_id str, slice str, task str, template_id str, split str, doc_id str,
  source int16, row_index int64 (document row), position int64 (target position = final token of the prompt),
  token_ids list<int32> (tokens 0..position), y_hat int32, p_y_hat float32, logp_ga float32 (= log p_y_hat),
  objective_value float32 (the attributed scalar: log p_y_hat, or the top-1 minus top-2 logit gap),
  top_ids list<int32>, top_logprobs list<float32> (clean next-token top-n), teacher_valid bool,
  idx list<uint16> [layers * k], s list<float32> [layers * k], h list<float16> [layers * k]   (layer-major, each
  layer's k entries in descending |s|), s_abs_sum list<float32> [layers], s_sum list<float32> [layers],
  share_at_k list<float32> [layers * len(SHARE_GRID)] (share of sum |s| in the top k, k in SHARE_GRID),
  k_at_share list<int16> [layers * 4] (smallest k reaching 50/80/90/95% of sum |s|),
  n_above_rel list<int16> [layers * 3] (neurons with |s| >= 0.1 / 0.01 / 0.001 x the layer's max |s|)

The parquet schema metadata ``nc_wide`` (JSON) holds the part's k, base_k, layers, blocks, per-block crosscheck values
(top-base_k and top-k overlap, gradient relative error) and timing. ``<out-dir>/run_meta.json`` records the model
(path, HF revision), tokenizer, code commit, library versions, device and arguments.

Why ``s`` is float32, not float16: gradient products underflow fp16 (the upstream stage stores float32 for this
reason), and float32 keeps ``|s|`` bit-identical to the upstream score. ``h`` is bf16 inside the model and is stored
as float16 (exact for bf16 values in fp16's normal range); non-finite casts are counted in the metadata.

  python -m neuron_communities attribution-wide --manifest-dir WORLD --out-dir HARVEST --k 50 --shard 0 --shards 64
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from neuron_communities import config
from neuron_communities.data.io import TokensReader, load_manifest_with_dataset, load_meta, source_names
from neuron_communities.harvest.attribution import argmax_logprob
from neuron_communities.tail import CachedTail, kv_bytes_per_row_token_layer, load_model

WIDE_K = 1000   # decided 2026-11-22: store the top 1,000 per layer (~259 KB/prompt on corpus prompts, ~544 GB for 2.1M).
# To cut storage later, prune in place: keep the top 50, or (recommended) max(top 50, every neuron with |s| >= 0.1 x the
# layer's max |s|), ~36 KB/prompt (~76 GB). Rows are in descending |s|, so both are prefixes of each layer's entries.
# truncation statistics stored per prompt and layer (cheap: one sort of the layer's scores)
SHARE_GRID = (5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000)   # share of sum |s| kept by the top k
SHARE_LEVELS = (0.5, 0.8, 0.9, 0.95)                             # smallest k keeping this share
REL_LEVELS = (0.1, 0.01, 0.001)                                  # neurons with |s| >= level * max |s|
META_FIELDS = ("prompt_id", "slice", "task", "template_id", "split", "doc_id")


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--manifest-dir", type=Path, required=True, help="directory with manifest.npz and meta.json")
    ap.add_argument("--tokens", type=Path, default=None, help="tokens dataset (default: meta.json 'dataset')")
    ap.add_argument("--out-dir", type=Path, required=True, help="harvest dir; parts go to <out-dir>/parts/")
    ap.add_argument("--k", type=int, default=WIDE_K, help="neurons kept per layer")
    ap.add_argument("--base-k", type=int, default=config.TOP_K, help="k of the upstream crosscheck reproduced in metadata")
    ap.add_argument("--top-logprobs", type=int, default=20, help="clean next-token log-probs stored per prompt")
    ap.add_argument("--split", default="", help="split label when prompts.parquet does not give one (e.g. train, heldout)")
    ap.add_argument("--slice", default="", help="slice label when prompts.parquet does not give one (default: source name)")
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--block-size", type=int, default=config.BLOCK_SIZE)
    ap.add_argument("--part-blocks", type=int, default=16, help="blocks per parquet part (resume granularity)")
    ap.add_argument("--model", default=config.MODEL_ID, help="HF id or local snapshot path (pin the revision via the path)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--kv-budget-gb", type=float, default=8.0)
    ap.add_argument("--forward-token-budget", type=int, default=49152)
    ap.add_argument("--max-parts", type=int, default=0, help="smoke/pilot: process at most this many parts of the shard")
    ap.add_argument("--part-range", default="", help="START:END, harvest parts START..END-1 instead of the --shard's "
                    "range (used by the claim-based full-run worker, scripts/full/worker.sh)")
    ap.add_argument("--objective", choices=("logp", "logit_diff"), default="logp",
                    help="logp: d log p(top-1)/dh (upstream); logit_diff: d (logit top-1 - logit top-2)/dh, which stays "
                         "informative when the softmax is saturated")
    ap.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16",
                    help="model dtype (bfloat16 as upstream; float32 for small models)")
    ap.add_argument("--crosscheck-max-tokens", type=int, default=0,
                    help="run the naive full-backward crosscheck only on rows with at most this many tokens (0: no cap, "
                         "the upstream choice). The check is a diagnostic; stored attributions do not depend on it")


# ---- provenance -----------------------------------------------------------------------------------------------------

def code_commit() -> dict:
    root = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                                    capture_output=True, text=True, check=True).stdout.strip())
    except (OSError, subprocess.CalledProcessError):
        # no .git (e.g. a copy synced to a cluster): the submitter writes .code_commit = "<sha> <dirty 0/1>"
        f = root / ".code_commit"
        if f.exists():
            sha, d = f.read_text().split()[:2]
            return {"commit": sha, "dirty": d == "1", "source": ".code_commit"}
        sha, dirty = "unknown", True
    return {"commit": sha, "dirty": dirty}


def model_revision(model, model_arg: str) -> str:
    rev = getattr(model.config, "_commit_hash", None)
    if rev:
        return str(rev)
    parts = Path(str(model_arg)).parts
    return parts[parts.index("snapshots") + 1] if "snapshots" in parts and parts.index("snapshots") + 1 < len(parts) else "unknown"


def write_run_meta(args, model, out_dir: Path) -> None:
    import transformers
    meta = {
        "stage": "attribution-wide", "objective": args.objective, "dtype": args.dtype,
        "score": ("s = h * d log p(top-1) / dh, ranked by |s| (upstream act_ga)" if args.objective == "logp"
                  else "s = h * d (logit top-1 - logit top-2) / dh, ranked by |s|"),
        "model": str(args.model), "model_revision": model_revision(model, args.model),
        "tokenizer": "the model's (prompts are pre-tokenised in the tokens dataset; see its provenance.json)",
        "k": args.k, "base_k": args.base_k, "top_logprobs": args.top_logprobs,
        "block_size": args.block_size, "part_blocks": args.part_blocks, "kv_budget_gb": args.kv_budget_gb,
        "forward_token_budget": args.forward_token_budget, "manifest_dir": str(args.manifest_dir),
        "manifest_meta": load_meta(args.manifest_dir) if (args.manifest_dir / "meta.json").exists() else {},
        "manifest_sha256": {f: hashlib.sha256((args.manifest_dir / f).read_bytes()).hexdigest()
                            for f in ("manifest.npz", "dedup_kept.npz", "dedup_drop.npz", "prompts.parquet", "meta.json")
                            if (args.manifest_dir / f).exists()},
        "code": code_commit(), "torch": torch.__version__, "transformers": transformers.__version__,
        "python": platform.python_version(), "device": args.device,
        "device_name": torch.cuda.get_device_name() if args.device.startswith("cuda") else platform.processor(),
        "crosscheck_seed": "np.random.default_rng(block * 131 + first row of the sub-batch), as upstream",
        "logprobs": f"clean next-token top-{args.top_logprobs} log-probs per prompt",
        "reasoning_trace": None, "reasoning_note": "base model, single forward/backward per prompt: no generation, no trace",
    }
    path = out_dir / f"run_meta_shard{args.shard:04d}.json"
    path.write_text(json.dumps(meta, indent=1, default=str))


# ---- prompt metadata ------------------------------------------------------------------------------------------------

def load_prompt_meta(manifest_dir: Path, n: int, split: str, slice_: str) -> dict[str, np.ndarray]:
    """Per-manifest-index metadata columns (object arrays of str) plus ``source`` (int16)."""
    with np.load(manifest_dir / "manifest.npz") as m:
        source = m["source"].astype(np.int16) if "source" in m.files else np.zeros(n, np.int16)
    meta = load_meta(manifest_dir) if (manifest_dir / "meta.json").exists() else {}
    names = source_names(meta)
    cols = {f: np.full(n, "", dtype=object) for f in META_FIELDS}
    cols["prompt_id"] = np.array([f"{manifest_dir.name}:{i}" for i in range(n)], dtype=object)
    cols["slice"] = np.array([slice_ or names.get(int(s), "all") for s in source], dtype=object)
    cols["split"][:] = split
    p = manifest_dir / "prompts.parquet"
    if p.exists():
        t = pq.read_table(p).to_pydict()
        mi = np.asarray(t["manifest_index"], np.int64)
        for f in META_FIELDS:
            if f in t:
                cols[f][mi] = np.asarray([str(x) for x in t[f]], dtype=object)
    cols["source"] = source
    return cols


# ---- the stage ------------------------------------------------------------------------------------------------------

def objective_fn(name: str):
    """Per-row (B,) scalar whose gradient is attributed: log p(top-1) (upstream) or logit(top-1) - logit(top-2)."""
    if name == "logp":
        return argmax_logprob

    def logit_diff(logits: torch.Tensor) -> torch.Tensor:
        top2 = torch.topk(logits.float(), 2, dim=-1).values
        return top2[..., 0] - top2[..., 1]
    return logit_diff


def part_schema(n_layers: int, k: int) -> pa.Schema:
    return pa.schema([
        ("manifest_index", pa.int64()), ("block", pa.int32()), *[(f, pa.string()) for f in META_FIELDS],
        ("source", pa.int16()), ("row_index", pa.int64()), ("position", pa.int64()), ("token_ids", pa.list_(pa.int32())),
        ("y_hat", pa.int32()), ("p_y_hat", pa.float32()), ("logp_ga", pa.float32()), ("objective_value", pa.float32()),
        ("top_ids", pa.list_(pa.int32())), ("top_logprobs", pa.list_(pa.float32())), ("teacher_valid", pa.bool_()),
        ("idx", pa.list_(pa.uint16())), ("s", pa.list_(pa.float32())), ("h", pa.list_(pa.float16())),
        ("s_abs_sum", pa.list_(pa.float32())), ("s_sum", pa.list_(pa.float32())),
        ("share_at_k", pa.list_(pa.float32())), ("k_at_share", pa.list_(pa.int16())), ("n_above_rel", pa.list_(pa.int16())),
    ])


def run(args: argparse.Namespace) -> None:
    DEV, K, K0 = args.device, args.k, args.base_k
    assert 1 <= K0 <= K, f"need 1 <= base_k ({K0}) <= k ({K})"

    man_row, man_pos, dataset = load_manifest_with_dataset(args.manifest_dir, args.tokens)
    n_man = len(man_row)
    n_blocks = (n_man + args.block_size - 1) // args.block_size
    n_parts = (n_blocks + args.part_blocks - 1) // args.part_blocks
    pps = (n_parts + args.shards - 1) // args.shards
    my_parts = list(range(args.shard * pps, min((args.shard + 1) * pps, n_parts)))
    if args.part_range:
        lo, hi = (int(x) for x in args.part_range.split(":"))
        assert 0 <= lo < hi, f"bad --part-range {args.part_range}"
        my_parts = list(range(lo, min(hi, n_parts)))
    out_dir = args.out_dir / "parts"
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = [p for p in my_parts if not (out_dir / f"part_{p:06d}.parquet").exists()]
    if args.max_parts:
        todo = todo[: args.max_parts]
    if not my_parts:
        print(f"shard {args.shard}: no parts (total {n_parts})", flush=True)
        return
    print(f"shard {args.shard}/{args.shards}: parts {my_parts[0]}..{my_parts[-1]} of {n_parts}, {len(todo)} to do", flush=True)
    if not todo:
        return

    pmeta = load_prompt_meta(args.manifest_dir, n_man, args.split, args.slice)
    tokens = TokensReader(dataset)
    if args.dtype == "bfloat16":
        model = load_model(args.model, DEV)                       # the upstream loader
    else:
        from transformers import AutoModelForCausalLM
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(DEV)
        model.eval().requires_grad_(False)
    objective = objective_fn(args.objective)
    write_run_meta(args, model, args.out_dir)
    n_layers = len(model.model.layers)
    layer0 = model.model.layers[0]
    tail = CachedTail(model, split_layer=0)
    tail_layers = [layer for _idx, layer in tail._layers]
    assert len(tail_layers) == n_layers - 1
    kv_bytes = kv_bytes_per_row_token_layer(model)
    check_layer = min(18, n_layers - 1)
    d_mlp = int(model.config.intermediate_size)
    if K > d_mlp:
        print(f"k={K} > d_mlp={d_mlp}: storing all {d_mlp} neurons per layer", flush=True)
        K = d_mlp
    assert d_mlp < 65536, "neuron indices are stored as uint16"
    schema = part_schema(n_layers, K)

    # ---- the next three functions are the upstream ones (attribution.py), with logits returned as well ----
    def batches(targets):
        order = np.argsort([p for _, p in targets])
        i = 0
        while i < len(order):
            sub = [order[i]]
            i += 1
            while i < len(order):
                s_max = targets[order[i]][1] + 1
                rows = len(sub) + 1
                if rows * s_max * (n_layers - 1) * kv_bytes > args.kv_budget_gb * 1e9 \
                        or rows * s_max > args.forward_token_budget:
                    break
                sub.append(order[i])
                i += 1
            docs = [tokens.doc_tokens(r, p + 2) for r, p in (targets[r] for r in sub)]
            toks = [d[: targets[r][1] + 1] for d, r in zip(docs, sub)]
            valid = torch.tensor([len(d) == len(t) + 1 for d, t in zip(docs, toks)], device=DEV)
            s_max = max(len(t) for t in toks)
            input_ids = torch.zeros(len(sub), s_max, dtype=torch.long)
            attn = torch.zeros(len(sub), s_max, dtype=torch.long)
            for j, t in enumerate(toks):
                input_ids[j, : len(t)] = t
                attn[j, : len(t)] = 1
            tpos = torch.tensor([len(t) - 1 for t in toks], device=DEV)
            yield sub, toks, input_ids.to(DEV), attn.to(DEV), tpos, valid

    def clean_capture(input_ids, attn, tpos):
        cap = {}
        h1 = layer0.register_forward_hook(lambda _m, _i, out: cap.__setitem__("rp0", out[0] if isinstance(out, tuple) else out))
        h2 = layer0.mlp.down_proj.register_forward_pre_hook(lambda _m, ins: cap.__setitem__("h0", ins[0]))
        try:
            with torch.no_grad():
                model.model(input_ids=input_ids, attention_mask=attn)
        finally:
            h1.remove(); h2.remove()
        rows = torch.arange(input_ids.shape[0], device=DEV)
        return cap["rp0"], cap["h0"][rows, tpos]

    def attribute(input_ids, attn, tpos):
        rp0, h0c = clean_capture(input_ids, attn, tpos)
        ctx = tail.precompute(rp0, None, attn, tpos)
        rows = torch.arange(input_ids.shape[0], device=DEV)
        with torch.enable_grad():
            h0 = h0c.detach().clone().requires_grad_(True)
            hidden0 = rp0[rows, tpos] - layer0.mlp.down_proj(h0c) + layer0.mlp.down_proj(h0)
            captured: list[torch.Tensor] = []
            handles = [layer.mlp.down_proj.register_forward_pre_hook(lambda _m, ins: captured.append(ins[0]))
                       for layer in tail_layers]
            try:
                logits = tail.step_logits(hidden0, ctx)
            finally:
                for hd in handles:
                    hd.remove()
            assert len(captured) == n_layers - 1, f"captured {len(captured)} tail MLP inputs"
            hs = [h0] + captured
            loss = objective(logits)
            g = torch.autograd.grad(loss.sum(), hs)
            grad = torch.stack([gi.reshape(gi.shape[0], -1).float() for gi in g])
        h = torch.stack([hi.detach().reshape(hi.shape[0], -1).float() for hi in hs])
        return h, grad, loss.detach(), logits.detach()

    def naive_crosscheck(input_ids_row, attn_row, tpos_row, h_ref, g_ref):
        """Upstream crosscheck; returns (top-base_k overlap, top-k overlap, max relative gradient error)."""
        cap: list[torch.Tensor] = []
        handles = [model.model.layers[L].mlp.down_proj.register_forward_pre_hook(lambda _m, ins: cap.append(ins[0]))
                   for L in range(n_layers)]
        try:
            with torch.enable_grad():
                emb = model.model.embed_tokens(input_ids_row)
                emb.requires_grad_(True)
                out = model.model(inputs_embeds=emb, attention_mask=attn_row)
                hid = out.last_hidden_state[0, tpos_row]
                logits = model.lm_head(hid)[None]
                loss = objective(logits)
                g = torch.autograd.grad(loss.sum(), cap)
        finally:
            for hd in handles:
                hd.remove()
        h_n = torch.stack([c.detach()[0, tpos_row].float() for c in cap])
        g_n = torch.stack([x[0, tpos_row].float() for x in g])
        rel = ((g_n - g_ref).norm(dim=-1) / (g_n.norm(dim=-1) + 1e-12)).max().item()
        sc_ref = h_ref[check_layer].abs() * g_ref[check_layer].abs()
        sc_n = h_n[check_layer].abs() * g_n[check_layer].abs()
        ov = []
        for kk in (K0, K):
            a = set(torch.topk(sc_ref, kk).indices.tolist()); b = set(torch.topk(sc_n, kk).indices.tolist())
            ov.append(len(a & b) / kk)
        return ov[0], ov[1], rel

    def process_block(b: int, rows_out: list[dict], blk_meta: list[dict]) -> int:
        lo, hi = b * args.block_size, min((b + 1) * args.block_size, n_man)
        targets = [(int(man_row[i]), int(man_pos[i])) for i in range(lo, hi)]
        recs: list[dict | None] = [None] * len(targets)
        checks0, checks, rels, n_tok, n_nonfinite_h, n_skipped = [], [], [], 0, 0, 0
        for sub, toks, input_ids, attn, tpos, valid in batches(targets):
            h, grad, loss, logits = attribute(input_ids, attn, tpos)
            score = h.abs() * grad.abs()                               # upstream score, (L, B, d_mlp)
            val, idx = torch.topk(score, K, dim=-1)
            s = torch.gather(h * grad, -1, idx)
            assert torch.equal(s.abs(), val), "|s| must equal the upstream |h| x |grad| score"
            hv = torch.gather(h, -1, idx)
            abs_sum, sgn_sum = score.sum(-1), (h * grad).sum(-1)       # (L, B)
            srt = torch.sort(score, dim=-1, descending=True).values
            frac = srt.cumsum(-1) / abs_sum.unsqueeze(-1).clamp_min(1e-30)
            share_k = frac[..., [k - 1 for k in SHARE_GRID if k <= d_mlp]]                       # (L, B, G)
            k_share = torch.stack([(frac < lv).sum(-1) + 1 for lv in SHARE_LEVELS], -1).clamp_max(d_mlp)
            n_rel = torch.stack([(srt >= lv * srt[..., :1]).sum(-1) for lv in REL_LEVELS], -1)
            share_c, kshare_c, nrel_c = share_k.cpu().numpy(), k_share.cpu().numpy(), n_rel.cpu().numpy()
            del srt, frac
            lp = torch.log_softmax(logits.float(), dim=-1)
            top_lp, top_id = torch.topk(lp, args.top_logprobs, dim=-1)
            obj_val = loss.float().cpu().numpy()                   # the attributed scalar (= log p(top-1) for logp)
            h16 = hv.to(torch.float16)
            n_nonfinite_h += int((~torch.isfinite(h16)).sum())
            idx_c, s_c, h_c = idx.cpu().numpy(), s.cpu().numpy(), h16.cpu().numpy()
            abs_c, sgn_c = abs_sum.cpu().numpy(), sgn_sum.cpu().numpy()
            top_lp_c, top_id_c = top_lp.cpu().numpy(), top_id.cpu().numpy()
            for jj, r in enumerate(sub):
                mi = lo + int(r)
                recs[r] = {
                    "manifest_index": mi, "block": b, **{f: pmeta[f][mi] for f in META_FIELDS},
                    "source": int(pmeta["source"][mi]), "row_index": int(man_row[mi]), "position": int(man_pos[mi]),
                    "token_ids": toks[jj].to(torch.int32).numpy(), "y_hat": int(top_id_c[jj, 0]),
                    "p_y_hat": float(np.exp(top_lp_c[jj, 0])), "logp_ga": float(top_lp_c[jj, 0]), "objective_value": float(obj_val[jj]),
                    "top_ids": top_id_c[jj].astype(np.int32), "top_logprobs": top_lp_c[jj].astype(np.float32),
                    "teacher_valid": bool(valid[jj]),
                    "idx": idx_c[:, jj].astype(np.uint16).reshape(-1), "s": s_c[:, jj].astype(np.float32).reshape(-1),
                    "h": h_c[:, jj].reshape(-1), "s_abs_sum": abs_c[:, jj].astype(np.float32),
                    "s_sum": sgn_c[:, jj].astype(np.float32),
                    "share_at_k": share_c[:, jj].astype(np.float32).reshape(-1),
                    "k_at_share": kshare_c[:, jj].astype(np.int16).reshape(-1),
                    "n_above_rel": nrel_c[:, jj].astype(np.int16).reshape(-1),
                }
                n_tok += len(toks[jj])
            rng = np.random.default_rng(b * 131 + int(sub[0]))
            if args.crosscheck_max_tokens:
                ok = [jj for jj in range(len(sub)) if int(tpos[jj]) + 1 <= args.crosscheck_max_tokens]
                j = ok[int(rng.integers(len(ok)))] if ok else -1
            else:
                j = int(rng.integers(len(sub)))
            if j >= 0:
                ov0, ov, rel = naive_crosscheck(input_ids[j: j + 1, : tpos[j] + 1], attn[j: j + 1, : tpos[j] + 1], tpos[j],
                                                h[:, j], grad[:, j])
                checks0.append(ov0); checks.append(ov); rels.append(rel)
            else:
                n_skipped += 1
            del h, grad, score, val, idx, s, hv
            if DEV.startswith("cuda"):
                torch.cuda.empty_cache()
        assert all(r is not None for r in recs)
        rows_out.extend(recs)
        blk_meta.append({"block": b, "manifest_lo": lo, "n": hi - lo, "crosscheck": checks0, "crosscheck_k": checks,
                         "crosscheck_rel_err": rels, "nonfinite_h": n_nonfinite_h, "crosscheck_skipped": n_skipped})
        return n_tok

    for p in todo:
        t0 = time.monotonic()
        if DEV.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        rows_out: list[dict] = []
        blk_meta: list[dict] = []
        n_tok = 0
        for b in range(p * args.part_blocks, min((p + 1) * args.part_blocks, n_blocks)):
            n_tok += process_block(b, rows_out, blk_meta)
        secs = time.monotonic() - t0
        pmeta_json = {"part": p, "k": K, "base_k": K0, "layers": n_layers, "d_mlp": d_mlp, "blocks": blk_meta,
                      "share_grid": [k for k in SHARE_GRID if k <= d_mlp], "share_levels": list(SHARE_LEVELS),
                      "rel_levels": list(REL_LEVELS),
                      "seconds": secs, "prompts": len(rows_out), "tokens": n_tok, "check_layer": check_layer,
                      "peak_mem_gib": torch.cuda.max_memory_allocated() / 2**30 if DEV.startswith("cuda") else None,
                      "crosscheck_max_tokens": args.crosscheck_max_tokens, "objective": args.objective, "dtype": args.dtype}
        table = pa.Table.from_pylist(rows_out, schema=schema).replace_schema_metadata({"nc_wide": json.dumps(pmeta_json)})
        tmp = out_dir / f".part_{p:06d}.tmp.parquet"
        pq.write_table(table, tmp, compression="zstd")
        os.replace(tmp, out_dir / f"part_{p:06d}.parquet")
        cc = np.concatenate([m["crosscheck"] for m in blk_meta] + [[np.nan]]); rr = np.concatenate([m["crosscheck_rel_err"] for m in blk_meta] + [[np.nan]])
        print(f"part {p}: {len(rows_out)} prompts, {n_tok} tokens in {secs:.1f}s ({len(rows_out) / secs * 3600:.0f} prompts/h); "
              f"crosscheck top{K0} overlap {np.nanmean(cc):.3f}, grad rel err max {np.nanmax(rr):.2e} "
              f"({sum(m['crosscheck_skipped'] for m in blk_meta)} sub-batches skipped by the length cap); "
              f"non-finite h {sum(m['nonfinite_h'] for m in blk_meta)}; peak mem {pmeta_json['peak_mem_gib']}", flush=True)
    print(f"shard {args.shard} done", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
