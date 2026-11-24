"""Export a wide harvest for sharing: pruned attributions + prompt metadata, as Hugging Face-ready Parquet shards.

Pruning (decided 2026-11-22, the recommended rule): in every layer keep max(top 50, every neuron with
|s| >= 0.1 x the layer's max |s|), at most the stored k. Each layer's entries stay in descending |s|, so the pruned
entries are a prefix of the stored ones. ``idx`` / ``s`` / ``h`` become variable-length (layer-major) with
``n_kept`` (int16 per layer) giving each layer's count; every other harvest column is kept as is, including the
per-layer truncation statistics (computed over all 14,336 neurons at harvest time). Prompt-metadata columns of
``<prompts>/prompts.parquet`` that the harvest does not carry (task, template_id, text, answer, answer_token, ...)
are joined by manifest_index.

  python scripts/export/export_shared.py --harvest H --prompts P --out OUT/data/<config> --split train

Writes ``<out>/<split>-NNNNN.parquet`` (zstd) with about --rows-per-file rows each, and ``<out>/<split>_export.json``
(row counts, kept-entries statistics, the rule).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuron_communities.harvest.wide_blocks import part_files, part_meta              # noqa: E402

MIN_K, REL = 50, 0.1


def prune_table(t: pa.Table, L: int, K: int) -> pa.Table:
    n = t.num_rows
    idx = t.column("idx").combine_chunks().values.to_numpy().reshape(n, L, K)
    s = t.column("s").combine_chunks().values.to_numpy().reshape(n, L, K)
    h = t.column("h").combine_chunks().values.to_numpy(zero_copy_only=False).reshape(n, L, K)
    a = np.abs(s)
    keep = np.maximum(MIN_K, (a >= REL * a[:, :, :1]).sum(2)).clip(max=K).astype(np.int16)   # (n, L); a[:,:,0] is the max
    mask = np.arange(K)[None, None, :] < keep[:, :, None]
    counts = keep.sum(1).astype(np.int64)
    offs = np.concatenate([[0], np.cumsum(counts)]).astype(np.int32)
    cols = {}
    for name, arr, typ in (("idx", idx, pa.uint16()), ("s", s, pa.float32()), ("h", h, pa.float16())):
        cols[name] = pa.ListArray.from_arrays(pa.array(offs), pa.array(arr[mask], type=typ))
    t = t.drop_columns(["idx", "s", "h"])
    for name in ("idx", "s", "h"):
        t = t.append_column(name, cols[name])
    lo = np.arange(n + 1, dtype=np.int32) * L
    t = t.append_column("n_kept", pa.ListArray.from_arrays(pa.array(lo), pa.array(keep.ravel(), type=pa.int16())))
    return t


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, default=None, help="prompt/manifest dir whose prompts.parquet to join")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--rows-per-file", type=int, default=10000)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    parts = part_files(a.harvest)
    m0 = part_meta(parts[0]); L, K = m0["layers"], m0["k"]
    pmeta = None
    if a.prompts and (a.prompts / "prompts.parquet").exists():
        pmeta = pq.read_table(a.prompts / "prompts.parquet")
    buf, nbuf, fileno, total, kept_sum = [], 0, 0, 0, np.zeros(L)
    def flush():
        nonlocal buf, nbuf, fileno
        if buf:
            pq.write_table(pa.concat_tables(buf), a.out / f"{a.split}-{fileno:05d}.parquet", compression="zstd",
                           row_group_size=1000)
            fileno += 1; buf, nbuf = [], 0
    for p in parts:
        t = pq.read_table(p)
        t = t.replace_schema_metadata(None)
        t = prune_table(t, L, K)
        if pmeta is not None:
            extra = [c for c in pmeta.column_names if c not in t.column_names and c != "manifest_index"]
            if extra:
                mi = t.column("manifest_index").to_numpy()
                pos = np.searchsorted(pmeta.column("manifest_index").to_numpy(), mi)
                sub = pmeta.take(pa.array(pos))
                assert np.array_equal(sub.column("manifest_index").to_numpy(), mi)
                for c in extra:
                    t = t.append_column(c, sub.column(c))
        kept_sum += np.stack(t.column("n_kept").to_numpy(zero_copy_only=False)).sum(0) if t.num_rows else 0
        total += t.num_rows
        buf.append(t); nbuf += t.num_rows
        if nbuf >= a.rows_per_file:
            flush()
    flush()
    info = {"rows": total, "files": fileno, "layers": L, "stored_k": K, "rule": f"max(top {MIN_K}, |s| >= {REL} x layer max)",
            "mean_kept_per_layer": [round(float(x / max(total, 1)), 1) for x in kept_sum], "source_harvest": str(a.harvest),
            "source_run_meta": sorted(str(x.name) for x in a.harvest.glob("run_meta_shard*.json"))}
    (a.out / f"{a.split}_export.json").write_text(json.dumps(info, indent=1))
    print(json.dumps({k: info[k] for k in ("rows", "files", "rule")}), "mean kept/layer:", round(float(kept_sum.sum() / max(total, 1) / L), 1))


if __name__ == "__main__":
    main()
