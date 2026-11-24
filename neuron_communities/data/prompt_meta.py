"""``prompt-meta`` stage (this fork): per-prompt identity of a manifest dir -> ``<manifest>/prompts.parquet``.

One row per manifest entry::

  manifest_index int64, prompt_id str ("<label>:<manifest_index>", stable across subsets of this manifest),
  split str, slice str (the source corpus tag), doc_id str ("<tag>:<id>" from the tokens dataset's docs.parquet),
  row_index int64, position int64 (offset of the target token in its document)

``attribution-wide`` joins these fields into every harvested row. Works on a ``combine``d manifest (one docs.parquet per
source, found through each source manifest's tokens dataset) and on a single-corpus manifest (``meta.dataset``).
Documents tokenised before docs.parquet existed get an empty doc_id (counted in the output).

  python -m neuron_communities prompt-meta --manifest-dir train/base --label train --split train
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from neuron_communities.data import io


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--manifest-dir", type=Path, required=True)
    ap.add_argument("--label", required=True, help="prompt_id prefix, e.g. train or heldout")
    ap.add_argument("--split", required=True, help="split label stored per prompt")


def doc_ids(dataset: Path, n_rows: int) -> np.ndarray:
    p = Path(dataset) / "docs.parquet"
    out = np.full(n_rows, "", dtype=object)
    if p.exists():
        t = pq.read_table(p, columns=["row", "doc_id"]).to_pydict()
        out[np.asarray(t["row"], np.int64)] = t["doc_id"]
    return out


def run(a: argparse.Namespace) -> None:
    md = a.manifest_dir
    meta = io.load_meta(md)
    with np.load(md / "manifest.npz") as m:
        row, pos = m["row_index"].astype(np.int64), m["position"].astype(np.int64)
        source = m["source"].astype(np.int64) if "source" in m.files else np.zeros(len(row), np.int64)
    n = len(row)
    names = io.source_names(meta)
    doc = np.full(n, "", dtype=object)
    srcs = meta.get("sources") if isinstance(meta.get("sources"), dict) else None
    if srcs and all("row_offset" in v and "manifest" in v for v in srcs.values()):
        for tag, v in srcs.items():
            off, nd = int(v["row_offset"]), int(v["n_docs"])
            ids = doc_ids(io.dataset_of(v["manifest"]), nd)
            sel = (row >= off) & (row < off + nd)
            doc[sel] = [f"{tag}:{d}" if d else "" for d in ids[row[sel] - off]]
    else:
        ds = io.dataset_of(md)
        ids = doc_ids(ds, len(io.read_counts(ds)))
        tag = names.get(0, "all")
        doc[:] = [f"{tag}:{d}" if d else "" for d in ids[row]]
    t = pa.table({
        "manifest_index": pa.array(np.arange(n, dtype=np.int64)),
        "prompt_id": pa.array([f"{a.label}:{i}" for i in range(n)], pa.string()),
        "split": pa.array([a.split] * n, pa.string()),
        "slice": pa.array([names.get(int(s), "all") for s in source], pa.string()),
        "doc_id": pa.array(list(doc), pa.string()),
        "row_index": pa.array(row), "position": pa.array(pos)})
    pq.write_table(t, md / "prompts.parquet")
    missing = int(sum(1 for d in doc if not d))
    print(f"prompts.parquet: {n:,} prompts, {missing:,} without a doc id -> {md / 'prompts.parquet'}", flush=True)
