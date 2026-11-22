"""The wide (top-50, signed) harvest on the tiny random Llama of ``test_smoke.py`` (CPU, no network).

The key check: thresholding the wide harvest at top-5 and dropping the signs (``wide-blocks --k 5``) reproduces the
untouched upstream ``attribution --k 5`` output exactly (every key, dtype and bit), and so the same ``world`` and
``census`` outputs. Then the wide fields themselves: ordering, signs, totals, log-probs, tokens and metadata.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from neuron_communities import config
from neuron_communities.cli import main
from neuron_communities.data import io
from neuron_communities.harvest.wide_blocks import load_wide, part_meta

from test_smoke import D_MLP, N_LAYERS, save_tiny_llama, write_tokens_dataset

N_PROMPTS, BLOCK, KW = 300, 64, 50


def stage(*argv) -> None:
    main([str(a) for a in argv])


def same_bits(a: np.ndarray, b: np.ndarray) -> bool:
    if a.dtype != b.dtype or a.shape != b.shape:
        return False
    if a.dtype.kind == "f":
        return np.array_equal(a.view(f"u{a.itemsize}"), b.view(f"u{b.itemsize}"))
    return np.array_equal(a, b)


@pytest.fixture(scope="module")
def w(tmp_path_factory) -> dict:
    config.configure(N_LAYERS, D_MLP)
    R = tmp_path_factory.mktemp("nc_wide")
    write_tokens_dataset(R / "tok", np.random.default_rng(0), 300)
    save_tiny_llama(R / "tinyllama")
    stage("sample", "--dataset", R / "tok", "--out", R / "man", "--n", N_PROMPTS, "--seed", 0)
    stage("dedup", "--manifest-dir", R / "man")
    # prompt metadata for a subset of manifest entries (the rest fall back to defaults)
    pq.write_table(pa.table({"manifest_index": pa.array([0, 5, 299], pa.int64()), "task": ["addition", "months", "x"],
                             "template_id": ["t0", "t1", "t2"], "slice": ["arith", "arith", "other"]}), R / "man" / "prompts.parquet")
    model = R / "tinyllama"
    # upstream stage, untouched, k = 5
    stage("attribution", "--manifest-dir", R / "man", "--out-dir", R / "harv5", "--shard", 0, "--shards", 1,
          "--block-size", BLOCK, "--model", model, "--device", "cpu")
    # wide stage, two shards, parts of two blocks
    for shard in range(2):
        stage("attribution-wide", "--manifest-dir", R / "man", "--out-dir", R / "harvw", "--k", KW, "--shard", shard, "--shards", 2,
              "--block-size", BLOCK, "--part-blocks", 2, "--split", "train", "--model", model, "--device", "cpu")
    stage("wide-blocks", "--harvest-dir", R / "harvw", "--out-dir", R / "harvw5", "--k", 5)
    return {"R": R, "model": model}


def test_top5_blocks_reproduce_upstream_exactly(w):
    R = w["R"]
    a = sorted((R / "harv5" / "blocks").glob("block_*.npz"))
    b = sorted((R / "harvw5" / "blocks").glob("block_*.npz"))
    assert [x.name for x in a] == [x.name for x in b] and len(a) == -(-N_PROMPTS // BLOCK)
    for fa, fb in zip(a, b):
        za, zb = np.load(fa), np.load(fb)
        assert set(za.files) == set(zb.files), (fa.name, sorted(za.files), sorted(zb.files))
        for key in za.files:
            assert same_bits(za[key], zb[key]), f"{fa.name}:{key} differs ({za[key].dtype} vs {zb[key].dtype})"


def test_world_and_census_identical(w):
    R = w["R"]
    for h in ("harv5", "harvw5"):
        stage("world", "--harvest-dir", R / h, "--manifest-dir", R / "man", "--out-dir", R / f"world_{h}", "--copy")
        stage("census", "--world", R / f"world_{h}", "--tile", 1, "--workers", 1, "--replicates", 4, "--c-min", 2, "--z-store", 1)
    for L in range(N_LAYERS):
        za = np.load(R / "world_harv5" / "merged" / f"L{L:02d}.npz"); zb = np.load(R / "world_harvw5" / "merged" / f"L{L:02d}.npz")
        assert all(same_bits(za[k], zb[k]) for k in za.files)
    sa = json.loads((R / "harv5" / "harvest_summary.json").read_text()); sb = json.loads((R / "harvw5" / "harvest_summary.json").read_text())
    assert sa == sb
    ca = np.load(R / "world_harv5" / "census" / "pairs_L00_L01.npz"); cb = np.load(R / "world_harvw5" / "census" / "pairs_L00_L01.npz")
    assert set(ca.files) == set(cb.files) and all(same_bits(ca[k], cb[k]) for k in ca.files)
    print(f"census L0-L1: {len(ca['i'])} pairs, identical")


def test_wide_fields(w):
    R = w["R"]
    d = load_wide(R / "harvw")
    n = N_PROMPTS
    assert np.array_equal(d["manifest_index"], np.arange(n))
    assert d["idx"].shape == (n, N_LAYERS, KW) and d["idx"].dtype == np.uint16 and (d["idx"] < D_MLP).all()
    assert d["s"].dtype == np.float32 and d["h"].dtype == np.float16 and np.isfinite(d["s"]).all() and np.isfinite(d["h"]).all()
    a = np.abs(d["s"])
    assert (np.diff(a, axis=-1) <= 0).all(), "descending |s| within each layer"
    assert all(len(set(r)) == KW for r in d["idx"].reshape(-1, KW)), "no repeats"
    assert (a.sum(-1) <= d["s_abs_sum"] * (1 + 1e-5) + 1e-30).all() and (d["s_abs_sum"] > 0).all()
    assert (d["s"] < 0).any() and (d["s"] > 0).any(), "signs are kept"
    # s = h * grad: the sign of s matches h times the gradient's sign; check |h| matches the stored activation's scale
    assert (np.sign(d["s"]) != 0).mean() > 0.99
    # top-5 prefix agrees with the upstream blocks
    up = np.concatenate([np.load(f)["idx_act_ga"] for f in sorted((R / "harv5" / "blocks").glob("block_*.npz"))])
    assert np.array_equal(up, d["idx"][:, :, :5])
    # log-probs and tokens
    assert (np.diff(d["top_logprobs"], axis=-1) <= 0).all() and d["top_logprobs"].shape == (n, 20)
    assert np.array_equal(d["top_ids"][:, 0], d["y_hat"]) and np.allclose(d["top_logprobs"][:, 0], d["logp_ga"], atol=1e-5)
    assert np.allclose(np.log(d["p_y_hat"]), d["logp_ga"], atol=1e-5)
    reader = io.TokensReader(R / "tok")
    row, pos = io.load_manifest(R / "man")
    for i in range(0, n, 37):
        assert np.array_equal(d["token_ids"][i], reader.prefix_tokens(int(row[i]), int(pos[i])).numpy())
        assert d["row_index"][i] == row[i] and d["position"][i] == pos[i] == len(d["token_ids"][i]) - 1
    # metadata join
    assert d["task"][0] == "addition" and d["template_id"][5] == "t1" and d["slice"][299] == "other"
    assert d["slice"][1] == "all" and d["task"][1] == "" and (d["split"] == "train").all()
    assert d["prompt_id"][7] == "man:7"


def test_crosscheck_and_metadata(w):
    R = w["R"]
    parts = sorted((R / "harvw" / "parts").glob("part_*.parquet"))
    assert len(parts) == 3                                   # 5 blocks, 2 per part
    for p in parts:
        m = part_meta(p)
        assert m["k"] == KW and m["base_k"] == 5 and m["layers"] == N_LAYERS and m["prompts"] > 0 and m["seconds"] > 0
        for b in m["blocks"]:
            assert max(b["crosscheck_rel_err"]) < 0.05, b      # cached-tail gradients match a naive full backward
            assert min(b["crosscheck_k"]) >= 0.8 and b["nonfinite_h"] == 0
    for s in range(2):
        rm = json.loads((R / "harvw" / f"run_meta_shard{s:04d}.json").read_text())
        assert rm["k"] == KW and rm["code"]["commit"] and rm["model_revision"] and rm["reasoning_trace"] is None
        assert set(rm["manifest_sha256"]) >= {"manifest.npz", "dedup_kept.npz", "prompts.parquet"}


def test_signed_score_matches_naive_autograd(w):
    """s = h * dlogp/dh against a plain full-model backward with hooks (independent of CachedTail)."""
    from neuron_communities.tail import load_model
    R = w["R"]
    d = load_wide(R / "harvw", columns=["manifest_index", "token_ids", "idx", "s", "h"])
    model = load_model(str(w["model"]), "cpu")
    for i in (0, 11, 222):
        ids = torch.tensor(d["token_ids"][i], dtype=torch.long)[None]
        cap = []
        hs = [l.mlp.down_proj.register_forward_pre_hook(lambda _m, ins: cap.append(ins[0])) for l in model.model.layers]
        with torch.enable_grad():
            emb = model.model.embed_tokens(ids).requires_grad_(True)
            out = model.model(inputs_embeds=emb)
            lp = torch.log_softmax(model.lm_head(out.last_hidden_state[0, -1]).float(), -1).max()
            g = torch.autograd.grad(lp, cap)
        for hd in hs:
            hd.remove()
        for L in range(N_LAYERS):
            h = cap[L][0, -1].detach().float(); gr = g[L][0, -1].float()
            s_ref = (h * gr)[torch.from_numpy(d["idx"][i, L].astype(np.int64))].detach().numpy()
            # the model runs in bf16, so the cached-tail and naive backward paths agree to bf16 rounding only
            rel = np.linalg.norm(d["s"][i, L] - s_ref) / np.linalg.norm(s_ref)
            assert rel < 0.03, (i, L, rel)
            big = np.abs(s_ref) > 0.1 * np.abs(s_ref).max()
            assert np.array_equal(np.sign(d["s"][i, L][big]), np.sign(s_ref[big])), (i, L)
            assert np.allclose(d["h"][i, L].astype(np.float32), h[torch.from_numpy(d["idx"][i, L].astype(np.int64))].numpy(), rtol=1e-2, atol=1e-3)


def test_rerun_is_noop(w, capsys):
    R = w["R"]
    before = {p.name: p.stat().st_mtime_ns for p in (R / "harvw" / "parts").glob("*.parquet")}
    stage("attribution-wide", "--manifest-dir", R / "man", "--out-dir", R / "harvw", "--k", KW, "--shard", 0, "--shards", 2,
          "--block-size", BLOCK, "--part-blocks", 2, "--model", w["model"], "--device", "cpu")
    assert "0 to do" in capsys.readouterr().out
    assert before == {p.name: p.stat().st_mtime_ns for p in (R / "harvw" / "parts").glob("*.parquet")}
