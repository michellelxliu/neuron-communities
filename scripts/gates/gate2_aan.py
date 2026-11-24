"""Gate 2: does Goodfire's a/an community reappear in the pilot's dictionary?

Reference community (paper): L2:5664, L5:3846, L8:13390, L12:14182, L13:3604, L14:11888 (6 neurons).
Pass criterion, fixed before the run: one community of the final (hub-stripped) dictionary holds >= 4 of the 6.
Also reports, for each reference neuron, whether it entered the census graph at all, and the best-matching communities
of the with-hubs dictionary.

  python scripts/gates/gate2_aan.py --world W --out gate2_aan.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

D_MLP = 14336
AAN = [(2, 5664), (5, 3846), (8, 13390), (12, 14182), (13, 3604), (14, 11888)]
PASS_MIN = 4


def load(p: Path) -> list[dict]:
    with gzip.open(p, "rt") as f:
        d = json.load(f)
    return d if isinstance(d, list) else d.get("candidates", d.get("communities", []))


def best(cands: list[dict], ref: set[int], top: int = 5) -> list[dict]:
    rows = []
    for i, c in enumerate(cands):
        m = set(int(x) for x in c["members"])
        hit = len(m & ref)
        if hit:
            rows.append({"index": i, "hits": hit, "size": len(m), "jaccard": round(hit / len(m | ref), 4),
                         "found": c.get("found", [])[:6],
                         "hit_neurons": sorted(f"L{g // D_MLP}:{g % D_MLP}" for g in m & ref),
                         "members": sorted(f"L{g // D_MLP}:{g % D_MLP}" for g in m)[:40]})
    return sorted(rows, key=lambda r: (-r["hits"], -r["jaccard"]))[:top]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--world", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    ref = {l * D_MLP + i for l, i in AAN}
    # frequency of each reference neuron in the pilot's top-5 sets
    kept = np.load(a.world / "dedup_kept.npz")["kept"]
    freq = {}
    for l, i in AAN:
        z = np.load(a.world / "merged" / f"L{l:02d}.npz")
        indptr, idx = z["indptr"], z["indices"]
        rows = np.repeat(np.arange(len(indptr) - 1), np.diff(indptr))
        freq[f"L{l}:{i}"] = int(np.isin(rows[idx == i], kept).sum())
    # did the neuron appear in any stored census pair
    in_graph = {f"L{l}:{i}": False for l, i in AAN}
    for p in sorted((a.world / "census").glob("pairs_L*_L*.npz")):
        la, lb = int(p.name[7:9]), int(p.name[11:13])
        z = np.load(p)
        for l, i in AAN:
            if l == la and (np.any(z["i"] == i) or (la == lb and np.any(z["j"] == i))):
                in_graph[f"L{l}:{i}"] = True
            if l == lb and la != lb and np.any(z["j"] == i + D_MLP):
                in_graph[f"L{l}:{i}"] = True
    res = {"reference": [f"L{l}:{i}" for l, i in AAN], "pass_min_hits": PASS_MIN, "n_prompts_kept": int(len(kept)),
           "top5_frequency": freq, "in_census_graph": in_graph}
    for name in ("candidates", "candidates_with_hubs"):
        p = a.world / name / "candidates.json.gz"
        if p.exists():
            c = load(p)
            res[name] = {"n_communities": len(c), "best": best(c, ref)}
    b = res.get("candidates", {}).get("best", [])
    res["max_hits"] = b[0]["hits"] if b else 0
    res["gate2"] = "PASS" if res["max_hits"] >= PASS_MIN else "FAIL"
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("gate2", "max_hits", "top5_frequency", "in_census_graph")}, indent=1))
    for r in b[:3]:
        print(r["hits"], r["size"], r["jaccard"], r["hit_neurons"], r["found"])


if __name__ == "__main__":
    main()
