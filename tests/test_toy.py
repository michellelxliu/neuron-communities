"""Toy model (Build B) on CPU: planted overlapping neuron sets, the matched null, a short training run, verification and
the co-attribution diagnostics end to end. Checks that the pieces fit together, not that a short run plants circuits."""
from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pytest
import torch

from neuron_communities.toy.tasks import ToyConfig, ToyWorld
from neuron_communities.toy.train import off_circuit_masks, train


def test_overlapping_sets():
    cfg = ToyConfig(overlap_rho=0.25)
    w = ToyWorld(cfg)
    n_sh = round(0.25 * cfg.block_size)
    for key, (L, ids) in w.shared.items():
        a, b = key.split("&")
        assert len(ids) == n_sh
        assert set(ids) == set(w.blocks[a][1]) & set(w.blocks[b][1])
        assert w.layer_of[a] == w.layer_of[b] == L
    for p, (L, ids) in w.blocks.items():
        assert len(ids) == cfg.block_size
        assert len(w.private[p][1]) == cfg.block_size - (n_sh if any(p in k.split("&") for k in w.shared) else 0)
    # private sets, shared regions and free neurons partition each layer
    for L in (0, 1):
        parts = [set(ids) for p, (L2, ids) in w.private.items() if L2 == L] + \
                [set(ids) for L2, ids in w.shared.values() if L2 == L] + [set(w.free[L])]
        assert sum(map(len, parts)) == cfg.d_mlp and set().union(*parts) == set(range(cfg.d_mlp))
    # a task's circuit is the union of its two primitives' sets
    for k, (s, a) in enumerate(cfg.tasks):
        tn = w.task_neurons(k)
        assert set(tn[0]) == set(w.blocks[s][1]) and set(tn[1]) == set(w.blocks[a][1])
    off = off_circuit_masks(w)
    assert all(int((off[L][k] == 0).sum()) == cfg.block_size for L in (0, 1) for k in range(len(cfg.tasks)))


def test_null_shares_nothing():
    cfg = ToyConfig.null_of(ToyConfig(overlap_rho=0.25))
    w = ToyWorld(cfg)
    assert cfg.overlap_rho == 0 and all(not ids for _, ids in w.shared.values())
    assert (w.M.sum(0) == 1).all(), "every primitive is used by exactly one task"
    assert len(cfg.tasks) == len(ToyConfig().tasks)
    rt = ToyConfig.from_json(cfg.to_json())
    assert rt == cfg


def test_sample_answers():
    w = ToyWorld(ToyConfig())
    b = w.sample(500, np.random.default_rng(0))
    for tok, k, x, y in zip(b["tokens"], b["task"], b["x"], b["y"]):
        assert tok[-1] == w.eq_tok and tok[-2] == x and tok[-3] == w.task_tok[int(k)]
        assert y == w.answer(int(k), int(x))



def test_train_verify_diagnose(tmp_path):
    from neuron_communities.toy.diagnostics import diagnose
    from neuron_communities.toy.verify import verify
    torch.manual_seed(0)
    cfg = ToyConfig(name="t", overlap_rho=0.25, d_model=64, d_mlp=256, block_size=16, steps_mask=60, steps_finetune=30,
                    batch=64)
    run = tmp_path / "toy"
    meta = train(cfg, run, log_every=1000)
    assert {"model", "config.json", "ground_truth.json", "train_meta.json"} <= {p.name for p in run.iterdir()}
    assert meta["log"][-1]["phase"] == "finetune"
    v = verify(run, n_per_task=50)
    assert len(v["clean_train"]) == len(cfg.tasks)
    assert (run / "verification.md").exists()
    d = diagnose(run, n=200, ks=(5,))
    m = d["by_k"]["5"]
    assert {"modularity_raw", "modularity_z_pos", "density_raw", "density_z", "shared"} <= set(m)
    assert m["shared"]["n"] == 3 * round(0.25 * 16)
    assert json.loads((run / "diagnostics_logp.json").read_text())["n_prompts"] == d["n_prompts"]
