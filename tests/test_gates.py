"""Gate tooling on the tiny random Llama (CPU): prompt worlds, the gate-1 script, the crosscheck cap, the probe summary."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq
import pytest

from neuron_communities import config
from neuron_communities.cli import main
from neuron_communities.data import io
from neuron_communities.harvest.wide_blocks import load_wide, part_meta
from neuron_communities.slices import arithmetic as ar
from neuron_communities.slices.prompts import write_prompt_world

from test_smoke import D_MLP, N_LAYERS, save_tiny_llama


def stage(*argv) -> None:
    main([str(a) for a in argv])

ROOT = Path(__file__).resolve().parents[1]


def tiny_tokenizer(path: Path):
    """Byte-level BPE with a vocabulary that fits the tiny model (512)."""
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, processors, trainers
    from transformers import PreTrainedTokenizerFast
    tk = Tokenizer(models.BPE()); tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False); tk.decoder = decoders.ByteLevel()
    corpus = [r["text"] + r["answer"] for t in ar.TASKS for r in ar.make_prompts(t, 50, seed=1)]
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=400, special_tokens=["<pad>", "<s>", "</s>"],
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", tk.token_to_id("<s>"))])
    t = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", pad_token="<pad>")
    t.save_pretrained(path)
    return t


@pytest.fixture(scope="module")
def g(tmp_path_factory) -> dict:
    config.configure(N_LAYERS, D_MLP)
    R = tmp_path_factory.mktemp("nc_gates")
    save_tiny_llama(R / "tinyllama")
    tok = tiny_tokenizer(R / "tinyllama")
    assert max(tok.get_vocab().values()) < 512
    return {"R": R, "model": R / "tinyllama", "tok": tok}


def test_arithmetic_prompts():
    for t in ar.TASKS:
        p = ar.make_prompts(t, 256, seed=0)
        assert len(p) == min(256, len(ar.TASKS[t][0]) * len(ar.TASKS[t][1])) and len({r["text"] for r in p}) == len(p)
        assert p == ar.make_prompts(t, 256, seed=0), "fixed seed"
    r = ar.make_prompts("months", 1, pairs=[("November", "three")])[0]
    assert r["text"] == "Q: What month is three months after November?\nA:" and r["answer"] == " February"
    assert ar.make_prompts("hours", 1, pairs=[("22", "five")])[0]["answer"] == "03:00"
    assert ar.make_prompts("weekdays", 1, pairs=[("Sunday", "fourteen")])[0]["answer"] == " Sunday"
    assert ar.make_prompts("addition", 1, pairs=[("99", "100")])[0]["text"] == "99+100=" and ar._WORDS[47] == "forty-eight"
    assert ar._WORDS[20] == "twenty-one" and len(set(ar._WORDS)) == 48
    assert set(ar.NEURONS["months"]) <= set(ar.N_ADD) and set(ar.NEURONS["weekdays"]) <= set(ar.NEURONS["months"])
    assert set(ar.NEURONS["hours"]) - set(ar.N_ADD) == {1658}
    assert [len(ar.NEURONS[t]) for t in ("addition", "hours", "months", "weekdays")] == [28, 26, 16, 15]


def test_prompt_world_and_cap(g):
    R, tok = g["R"], g["tok"]
    recs = [r for t in ("months", "addition") for r in ar.make_prompts(t, 40, seed=0)]
    W = write_prompt_world(recs, tok, R / "pw")
    row, pos = io.load_manifest(W)
    rd = io.TokensReader(io.dataset_of(W))
    for i in (0, 41, 79):
        assert rd.doc(int(row[i])).tolist() == tok(recs[i]["text"])["input_ids"] and pos[i] == len(rd.doc(int(row[i]))) - 1
    stage("attribution-wide", "--manifest-dir", W, "--out-dir", R / "pw_h", "--shard", 0, "--model", g["model"], "--device", "cpu",
          "--block-size", 16, "--part-blocks", 2, "--crosscheck-max-tokens", 12)
    d = load_wide(R / "pw_h")
    assert list(d["task"][:3]) == ["months"] * 3 and d["template_id"][0] == "default" and d["split"][0] == "gate1"
    assert d["prompt_id"][0] == recs[0]["prompt_id"]
    m = [part_meta(p) for p in sorted((R / "pw_h" / "parts").glob("*.parquet"))]
    # months prompts are > 12 tokens, additions are short: some sub-batches skip the check, some run it
    sk = sum(b["crosscheck_skipped"] for x in m for b in x["blocks"]); ran = sum(len(b["crosscheck"]) for x in m for b in x["blocks"])
    assert sk > 0 and ran > 0 and all(x["crosscheck_max_tokens"] == 12 for x in m)
    stage("wide-blocks", "--harvest-dir", R / "pw_h", "--out-dir", R / "pw_b")
    out = subprocess.run([str(x) for x in [sys.executable, ROOT / "scripts" / "gates" / "probe_summary.py", "--harvest", R / "pw_h", "--slices", 4]],
                         capture_output=True, text=True, check=True).stdout
    s = json.loads(out)
    assert s["prompts_per_hour_per_device"] > 0 and len(s["topk_share_of_abs_s_by_layer"]["mean"]) == N_LAYERS
    assert all(0 < v <= 1.0001 for v in s["topk_share_of_abs_s_by_layer"]["mean"])


def test_gate1_script(g):
    R = g["R"]
    nj = R / "neurons.json"
    nj.write_text(json.dumps({t: [1, 2, 3, 5, 8] for t in ar.TASKS}))
    env = dict(os.environ, NEURON_COMMUNITIES_GEOMETRY=f"{N_LAYERS}x{D_MLP}")
    subprocess.run([str(x) for x in [sys.executable, ROOT / "scripts" / "gates" / "gate1_arith.py", "--model", g["model"], "--out", R / "g1",
                    "--device", "cpu", "--n", 24, "--layer", 1, "--top", 10, "--neurons-json", nj]], check=True, env=env)
    res = json.loads((R / "g1" / "gate1.json").read_text())
    for t in ("addition", "hours", "months", "weekdays"):
        r = res["tasks"][t]
        assert r["n"] == 24 and 0 <= r["direct"]["top40"] <= 5 and 0 <= r["accuracy_top1"] <= 1
        assert r["harvest_vs_direct_top1_agree"] >= 0.9, "harvest and direct paths agree on the predicted token (random model: near-ties in bf16)"
        assert r["harvest_vs_direct_top40_overlap"] >= 0.7, "same neurons rank highest by both paths"
    assert (R / "g1" / "gate1.md").exists() and pq.read_table(R / "g1" / "prompts" / "prompts.parquet").num_rows == 96
