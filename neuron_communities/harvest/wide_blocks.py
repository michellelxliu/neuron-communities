"""Reader for the wide harvest, and the ``wide-blocks`` stage: its top-k' prefix in the upstream block layout.

``load_wide(harvest_dir)`` returns the whole wide harvest as arrays in manifest order (``idx`` (n, L, k) uint16,
``s`` (n, L, k) float32, ``h`` (n, L, k) float16, ``s_abs_sum`` / ``s_sum`` (n, L), plus the per-prompt scalar columns)
for the mixed-membership analyses; ``columns=`` restricts what is read.

``wide-blocks`` writes ``<out-dir>/blocks/block_XXXXXX.npz`` with exactly the keys, dtypes and values of the upstream
``attribution`` stage run with ``--k <k'>`` (default 5): ``idx_act_ga`` = the first k' indices, ``val_act_ga`` = their
``|s|`` (bit-identical to the upstream ``|h| x |grad|``), and the row/position/teacher/log-prob/crosscheck fields. Run
``world`` on that directory to get the upstream world, then the unchanged graph, Leiden and evaluation stages.

  python -m neuron_communities wide-blocks --harvest-dir HARVEST_WIDE --out-dir HARVEST_TOP5 --k 5
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

VARIANT = "act_ga"


def add_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--harvest-dir", type=Path, required=True, help="wide harvest dir (with parts/)")
    ap.add_argument("--out-dir", type=Path, required=True, help="block harvest dir to write; blocks go to <out-dir>/blocks/")
    ap.add_argument("--k", type=int, default=5, help="neurons per layer to keep (a prefix of the wide top-k)")


def part_files(harvest_dir: Path) -> list[Path]:
    parts = sorted((Path(harvest_dir) / "parts").glob("part_*.parquet"))
    assert parts, f"no parts under {Path(harvest_dir) / 'parts'}"
    return parts


def part_meta(path: Path) -> dict:
    return json.loads(pq.read_schema(path).metadata[b"nc_wide"])


def load_wide(harvest_dir: Path, columns: list[str] | None = None) -> dict[str, np.ndarray]:
    """Every part concatenated in manifest order; list columns of fixed length become (n, L, k) or (n, L) arrays."""
    parts = part_files(harvest_dir)
    m0 = part_meta(parts[0]); L, K = m0["layers"], m0["k"]
    tables = [pq.read_table(p, columns=columns) for p in parts]
    out: dict[str, np.ndarray] = {}
    for name in tables[0].column_names:
        cols = [t.column(name) for t in tables]
        if name in ("idx", "s", "h", "s_abs_sum", "s_sum"):
            flat = np.concatenate([c.combine_chunks().values.to_numpy(zero_copy_only=False) for c in cols])
            out[name] = flat.reshape(-1, L, K) if name in ("idx", "s", "h") else flat.reshape(-1, L)
        elif name in ("token_ids", "top_ids", "top_logprobs"):
            out[name] = np.array([np.asarray(x) for c in cols for x in c.to_pylist()], dtype=object) \
                if name == "token_ids" else np.stack([np.asarray(x) for c in cols for x in c.to_pylist()])
        else:
            out[name] = np.concatenate([c.to_numpy(zero_copy_only=False) for c in cols])
    if "manifest_index" in out:
        order = np.argsort(out["manifest_index"], kind="stable")
        out = {k: v[order] for k, v in out.items()}
    return out


def run(args: argparse.Namespace) -> None:
    out = args.out_dir / "blocks"
    out.mkdir(parents=True, exist_ok=True)
    k = args.k
    n_written = 0
    for p in part_files(args.harvest_dir):
        meta = part_meta(p)
        assert 1 <= k <= meta["k"], f"k={k} exceeds the harvested k={meta['k']}"
        if k != meta["base_k"]:
            print(f"note: crosscheck overlap in the blocks is top-{meta['base_k']} (harvest base_k), not top-{k}")
        t = pq.read_table(p, columns=["manifest_index", "block", "row_index", "position", "teacher_valid", "logp_ga", "idx", "s"])
        L, K = meta["layers"], meta["k"]
        mi = t.column("manifest_index").to_numpy(); blk = t.column("block").to_numpy()
        idx = t.column("idx").combine_chunks().values.to_numpy().reshape(-1, L, K)
        s = t.column("s").combine_chunks().values.to_numpy().reshape(-1, L, K)
        cols = {c: t.column(c).to_numpy(zero_copy_only=False) for c in ("row_index", "position", "teacher_valid", "logp_ga")}
        for bm in meta["blocks"]:
            b = bm["block"]
            sel = np.flatnonzero(blk == b)
            sel = sel[np.argsort(mi[sel])]
            lo = bm["manifest_lo"]
            assert len(sel) == bm["n"] and np.array_equal(mi[sel], np.arange(lo, lo + bm["n"])), f"block {b}: rows incomplete"
            payload = {"row_index": cols["row_index"][sel].astype(np.int64), "position": cols["position"][sel].astype(np.int64),
                       "manifest_lo": np.array([lo]), "k": np.array([k]), "variants": np.array([VARIANT]),
                       "crosscheck": np.array(bm["crosscheck"], np.float32),
                       "crosscheck_rel_err": np.array(bm["crosscheck_rel_err"], np.float32),
                       "teacher_valid": cols["teacher_valid"][sel].astype(bool), "logp_ga": cols["logp_ga"][sel].astype(np.float32),
                       f"idx_{VARIANT}": np.ascontiguousarray(idx[sel, :, :k]).astype(np.uint16),
                       f"val_{VARIANT}": np.abs(s[sel, :, :k]).astype(np.float32)}
            tmp = out / f".block_{b:06d}.tmp.npz"
            np.savez_compressed(tmp, **payload)
            os.replace(tmp, out / f"block_{b:06d}.npz")
            n_written += 1
    print(f"wide-blocks: {n_written} blocks of top-{k} written to {out}", flush=True)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_args(ap)
    run(ap.parse_args(argv))


if __name__ == "__main__":
    main()
