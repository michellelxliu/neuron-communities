"""How much of each layer's attribution mass does a top-k (or a variable-length rule) keep, and what does it cost?

Reads a wide harvest that has the truncation statistics (``share_at_k``, ``k_at_share``, ``n_above_rel``) and the
measured on-disk size of its parts, and reports per layer: the share of sum |s| kept at each grid k, the k needed for
50/80/90/95%, and the neuron counts above relative thresholds; then, for candidate storage rules, the mean number of
stored entries per prompt and the bytes per prompt (measured bytes per stored entry from the parts themselves).

  python scripts/gates/curve_summary.py --harvest H --out curve.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuron_communities.harvest.wide_blocks import load_wide, part_files, part_meta   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--harvest", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    parts = part_files(args.harvest); m0 = part_meta(parts[0])
    grid, levels, rel = m0["share_grid"], m0["share_levels"], m0["rel_levels"]
    K, L = m0["k"], m0["layers"]
    d = load_wide(args.harvest, columns=["share_at_k", "k_at_share", "n_above_rel", "token_ids"])
    n = len(d["share_at_k"])
    disk = sum(p.stat().st_size for p in parts)
    tok_bytes = sum(len(t) for t in d["token_ids"]) * 4
    per_entry = (disk - tok_bytes) / (n * L * K)            # compressed bytes per stored (idx, s, h) entry, incl. overheads
    sh, ks, nr = d["share_at_k"], d["k_at_share"], d["n_above_rel"]
    out = {
        "n_prompts": n, "layers": L, "harvest_k": K, "disk_bytes": disk, "bytes_per_prompt_on_disk": disk / n,
        "token_bytes_per_prompt": tok_bytes / n, "compressed_bytes_per_entry": per_entry,
        "share_at_k": {str(k): {"mean_over_layers": float(sh[:, :, i].mean()), "min_layer_mean": float(sh[:, :, i].mean(0).min()),
                                "by_layer_mean": sh[:, :, i].mean(0).round(4).tolist()} for i, k in enumerate(grid)},
        "k_at_share": {str(lv): {"median_over_prompt_layers": float(np.median(ks[:, :, i])), "mean": float(ks[:, :, i].mean()),
                                 "p90": float(np.quantile(ks[:, :, i], 0.9)),
                                 "by_layer_median": np.median(ks[:, :, i], 0).tolist()} for i, lv in enumerate(levels)},
        "n_above_rel": {str(r): {"median": float(np.median(nr[:, :, i])), "mean": float(nr[:, :, i].mean()),
                                 "p90": float(np.quantile(nr[:, :, i], 0.9))} for i, r in enumerate(rel)},
    }
    rules = {f"top{k}": np.full((n, L), k) for k in grid if k >= 50}
    for i, lv in enumerate(levels):
        rules[f"share{int(lv * 100)}"] = ks[:, :, i]
        rules[f"max(50, share{int(lv * 100)}) cap 1000"] = np.clip(ks[:, :, i], 50, 1000)
    for i, r in enumerate(rel):
        rules[f"rel{r:g} (min 50)"] = np.maximum(nr[:, :, i], 50)
    out["rules"] = {name: {"entries_per_prompt": float(v.sum(1).mean()),
                           "bytes_per_prompt": float(v.sum(1).mean() * per_entry + tok_bytes / n)} for name, v in rules.items()}
    s = json.dumps(out, indent=1)
    print(s)
    if args.out:
        args.out.write_text(s)


if __name__ == "__main__":
    main()
