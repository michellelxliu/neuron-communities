"""``subset`` stage (this fork): a manifest dir holding a subset of another manifest's deduplicated prompts.

Used to run a pilot on part of the training world (so the full harvest can skip those prompts later) and, with
``--exclude``, to build the remainder. The candidates are the parent's kept rows minus its ``dedup_drop`` rows. With
``--per-source TAG=N,...`` the first N candidates of each source are taken (the parent manifest order is a random
permutation per source, so this is a uniform sample of each source); without it, all candidates are taken. The result
is shuffled with ``--seed`` so contiguous harvest blocks and shards mix sources and lengths.

Writes ``<out>/manifest.npz`` (row_index, position, source, pattern, subcat), ``dedup_kept.npz`` (all rows),
``parent_index.npy`` (parent manifest index of every row), ``prompts.parquet`` (the parent's rows re-indexed, so
prompt_id and doc_id carry over) and ``meta.json`` (the parent's dataset and sources plus the subset's provenance).

  python -m neuron_communities subset --parent train/base --out pilot --per-source fineweb=50000,wiki=25000,finemath=25000
  python -m neuron_communities subset --parent train/base --out rest --exclude pilot
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from neuron_communities.data import io


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--parent", type=Path, required=True, help="manifest dir with prompts.parquet (run prompt-meta first)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-source", default="", help="TAG=N,... prompts per source (default: every candidate)")
    ap.add_argument("--exclude", type=Path, action="append", default=[], help="subset dir(s) whose prompts to leave out")
    ap.add_argument("--seed", type=int, default=0)


def run(a: argparse.Namespace) -> None:
    P = a.parent
    meta = io.load_meta(P)
    assert (P / "prompts.parquet").exists(), "run prompt-meta on the parent first"
    kept = io.load_kept(P)
    cand = np.delete(kept, io.load_drop(P))
    for ex in a.exclude:
        cand = np.setdiff1d(cand, np.load(ex / "parent_index.npy"), assume_unique=True)
    m = dict(np.load(P / "manifest.npz"))
    n_par = len(m["row_index"])
    source = m["source"].astype(np.int64) if "source" in m else np.zeros(n_par, np.int64)
    names = io.source_names(meta)
    if a.per_source:
        want = {k: int(v) for k, v in (x.split("=") for x in a.per_source.split(","))}
        ids = {t: s for s, t in names.items()}
        picks = []
        for tag, k in want.items():
            c = cand[source[cand] == ids[tag]]
            if len(c) < k:
                raise SystemExit(f"{tag}: {len(c):,} candidates < {k:,} requested")
            picks.append(c[:k])
        sel = np.concatenate(picks)
    else:
        sel = cand
    sel = sel[np.random.default_rng(a.seed).permutation(len(sel))]
    a.out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out / "manifest.npz", **{k: v[sel] for k, v in m.items()})
    np.savez_compressed(a.out / "dedup_kept.npz", kept=np.arange(len(sel), dtype=np.int64))
    np.save(a.out / "parent_index.npy", sel)
    t = pq.read_table(P / "prompts.parquet")
    t = t.take(pa.array(sel))
    t = t.set_column(t.schema.get_field_index("manifest_index"), "manifest_index", pa.array(np.arange(len(sel), dtype=np.int64)))
    t = t.append_column("parent_manifest_index", pa.array(sel.astype(np.int64)))
    pq.write_table(t, a.out / "prompts.parquet")
    counts = {names.get(int(s), "all"): int(c) for s, c in zip(*np.unique(source[sel], return_counts=True))}
    sub_meta = {k: v for k, v in meta.items() if k in ("dataset", "sources")}
    sub_meta.update({"n": int(len(sel)), "subset_of": str(P), "per_source": a.per_source, "seed": a.seed,
                     "exclude": [str(x) for x in a.exclude], "counts": counts,
                     "selection": "parent kept rows minus dedup_drop rows, first N per source in parent order, shuffled"})
    (a.out / "meta.json").write_text(json.dumps(sub_meta, indent=1))
    n_doc = int(pc.sum(pc.not_equal(t["doc_id"], "")).as_py())
    print(f"subset: {len(sel):,} prompts {counts}, {n_doc:,} with doc ids -> {a.out}", flush=True)
