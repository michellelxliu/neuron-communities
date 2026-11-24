"""Source doc ids through the data stages (this fork): tokens -> docs.parquet, combine -> prompt-meta -> subset.

Two local JSONL corpora with a tiny word-level tokenizer (no network), sampled, deduplicated and combined as in the
pipeline; then every prompt's doc_id must name the document its tokens came from, and a pilot subset plus its
``--exclude`` complement must partition the parent's deduplicated prompts.
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

import numpy as np
import pyarrow.parquet as pq

from neuron_communities.cli import main
from neuron_communities.data import io


def stage(*argv) -> None:
    main([str(a) for a in argv])


def tokenizer(path):
    from tokenizers import Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast
    words = [f"w{i}" for i in range(40)]
    vocab = {"<s>": 0, "</s>": 1, "<unk>": 2, **{w: 3 + i for i, w in enumerate(words)}}
    tk = Tokenizer(models.WordLevel(vocab, unk_token="<unk>")); tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.post_processor = processors.TemplateProcessing(single="<s> $A", special_tokens=[("<s>", 0)])
    PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<s>", eos_token="</s>", unk_token="<unk>").save_pretrained(path)
    return words


def test_doc_ids_through_subset(tmp_path):
    words = tokenizer(tmp_path / "tok")
    rng = np.random.default_rng(0)
    texts = {}
    for name, key in (("a", "id"), ("b", "url")):          # corpus b has no id field: doc_id falls back to the url
        d = tmp_path / f"corpus_{name}"; d.mkdir()
        with open(d / "docs.jsonl", "w") as f:
            for i in range(60):
                t = " ".join(rng.choice(words, size=int(rng.integers(20, 80))).tolist())
                texts[f"{name}{i}"] = t
                f.write(json.dumps({"text": t, key: f"{name}{i}"}) + "\n")
        stage("tokens", "--hf-dataset", d, "--n-docs", 50, "--min-chars", 20, "--max-length", 64, "--chunk-tokens", 800,
              "--shuffle-buffer", 16, "--seed", 1, "--model", tmp_path / "tok", "--out", tmp_path / f"tok_{name}")
        docs = pq.read_table(tmp_path / f"tok_{name}" / "docs.parquet").to_pydict()
        assert docs["row"] == list(range(50)) and all(docs["doc_id"])
        assert (all(u == "" for u in docs["url"]) if key == "id" else docs["url"] == docs["doc_id"])
        # the stored tokens are the tokenisation of the named document
        from transformers import AutoTokenizer
        tk = AutoTokenizer.from_pretrained(tmp_path / "tok")
        reader = io.TokensReader(tmp_path / f"tok_{name}")
        for r in (0, 17, 49):
            want = tk(texts[docs["doc_id"][r]], truncation=True, max_length=64)["input_ids"]
            assert reader.doc(r).tolist() == want
        stage("sample", "--dataset", tmp_path / f"tok_{name}", "--out", tmp_path / f"man_{name}", "--n", 400, "--seed", 2)
        stage("dedup", "--manifest-dir", tmp_path / f"man_{name}")
    base = tmp_path / "base"
    stage("combine", "--source", f"{tmp_path / 'man_a'}:a", "--source", f"{tmp_path / 'man_b'}:b",
          "--out", base, "--tokens-out", tmp_path / "tok_all")
    stage("dedup", "--manifest-dir", base, "--short-only")
    stage("prompt-meta", "--manifest-dir", base, "--label", "train", "--split", "train")

    pm = pq.read_table(base / "prompts.parquet").to_pydict()
    row, pos = io.load_manifest(base)
    assert pm["prompt_id"][:3] == ["train:0", "train:1", "train:2"] and set(pm["split"]) == {"train"}
    meta = io.load_meta(base)
    for i in range(0, len(row), 37):
        tag, raw = pm["doc_id"][i].split(":", 1)
        assert tag == pm["slice"][i] and raw.startswith(tag)
        r_local = row[i] - meta["sources"][tag]["row_offset"]
        docs = pq.read_table(tmp_path / f"tok_{tag}" / "docs.parquet").to_pydict()
        assert docs["doc_id"][r_local] == raw and pm["position"][i] == pos[i]

    pilot, rest = tmp_path / "pilot", tmp_path / "rest"
    stage("subset", "--parent", base, "--out", pilot, "--per-source", "a=100,b=60", "--seed", 5)
    stage("subset", "--parent", base, "--out", rest, "--exclude", pilot, "--seed", 5)
    pi, ri = np.load(pilot / "parent_index.npy"), np.load(rest / "parent_index.npy")
    cand = np.delete(io.load_kept(base), io.load_drop(base))
    assert len(np.intersect1d(pi, ri)) == 0 and np.array_equal(np.sort(np.concatenate([pi, ri])), np.sort(cand))
    assert io.load_meta(pilot)["counts"] == {"a": 100, "b": 60}
    # subset rows point at the same prompts and keep the parent's identity
    prow, ppos = io.load_manifest(pilot)
    assert np.array_equal(prow, row[pi]) and np.array_equal(ppos, pos[pi])
    sp = pq.read_table(pilot / "prompts.parquet").to_pydict()
    assert sp["manifest_index"] == list(range(160)) and sp["prompt_id"] == [f"train:{i}" for i in pi]
    assert sp["doc_id"] == [pm["doc_id"][i] for i in pi]
    # the pilot is readable as a world manifest: same tokens through the parent's combined dataset
    rd = io.TokensReader(io.dataset_segments(io.load_meta(pilot)))
    rb = io.TokensReader(io.dataset_segments(meta))
    for j in (0, 80, 159):
        assert rd.prefix_tokens(int(prow[j]), int(ppos[j])).tolist() == rb.prefix_tokens(int(row[pi[j]]), int(pos[pi[j]])).tolist()
