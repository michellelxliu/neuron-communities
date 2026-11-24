"""Exploratory slices summary from a wide harvest: per task and template, accuracy (top-1 == answer token) and, where
the prompt has a competing form, the share of prompts whose clean top-20 log-probs rank the answer above it (an answer
in the top 20 beats an absent competitor). For agree_a_an, Goodfire's a/an community neurons: share of prompts with the
neuron in its layer's top 5 / top 50, and its mean |s| rank (stored top-k only; 'absent' beyond it).

  python scripts/gates/explore_summary.py --harvest H --prompts P --out summary.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuron_communities.harvest.wide_blocks import load_wide                         # noqa: E402

AAN = [(2, 5664), (5, 3846), (8, 13390), (12, 14182), (13, 3604), (14, 11888)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    t = load_wide(a.harvest, columns=["manifest_index", "y_hat", "top_ids", "top_logprobs", "idx"])
    pm = pq.read_table(a.prompts / "prompts.parquet").to_pandas().set_index("manifest_index")
    mi = np.asarray(t["manifest_index"], np.int64)
    task = pm.loc[mi, "task"].to_numpy(); tid = pm.loc[mi, "template_id"].to_numpy()
    ans = pm.loc[mi, "answer_token"].astype(np.int64).to_numpy(); alt = pm.loc[mi, "alt_token"].astype(np.int64).to_numpy()
    y = np.asarray(t["y_hat"], np.int64); top = t["top_ids"].astype(np.int64)
    correct = y == ans
    def rank(tokens):                                         # position in the top-20, 99 when absent
        hit = top == tokens[:, None]
        return np.where(hit.any(1), hit.argmax(1), 99)
    ra, rb = rank(ans), rank(alt)
    prefer = (ra < rb) & (ra < 99)
    has_alt = alt >= 0
    res = {"n": int(len(mi)), "tasks": {}}
    for tk in sorted(set(task)):
        m = task == tk
        r = {"n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4), "answer_in_top20": round(float((ra[m] < 99).mean()), 4)}
        if has_alt[m].any():
            r["prefers_answer_over_alt"] = round(float(prefer[m & has_alt].mean()), 4)
        r["templates"] = {tp: {"n": int((m & (tid == tp)).sum()), "accuracy": round(float(correct[m & (tid == tp)].mean()), 4)}
                          for tp in sorted(set(tid[m]))}
        res["tasks"][tk] = r
    m = task == "agree_a_an"
    if m.any():
        idx = t["idx"][m]                                     # (n, L, k), descending |s|
        k = idx.shape[2]
        aan = {}
        for L, i in AAN:
            pos = np.where(idx[:, L, :] == i, np.arange(k)[None, :], k).min(1)
            aan[f"L{L}:{i}"] = {"in_top5": round(float((pos < 5).mean()), 4), "in_top50": round(float((pos < 50).mean()), 4),
                                "median_rank": (int(np.median(pos)) if (pos < k).mean() >= 0.5 else f">{k}")}
        res["aan_community"] = aan
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({tk: {kk: v[kk] for kk in v if kk != "templates"} for tk, v in res["tasks"].items()}, indent=1))
    if "aan_community" in res:
        print(json.dumps(res["aan_community"]))


if __name__ == "__main__":
    main()
