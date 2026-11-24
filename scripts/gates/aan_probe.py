"""Diagnostic: where do Goodfire's a/an neurons rank on the a/an prompts, in every layer (to catch an indexing or
layer-numbering mismatch), and which neurons lead the a/an prompts by top-5 frequency per layer.

  python scripts/gates/aan_probe.py --harvest H --prompts P --out aan_probe.json
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
    t = load_wide(a.harvest, columns=["manifest_index", "y_hat", "idx", "s", "h"])
    pm = pq.read_table(a.prompts / "prompts.parquet").to_pandas().set_index("manifest_index")
    mi = np.asarray(t["manifest_index"], np.int64)
    task = pm.loc[mi, "task"].to_numpy()
    ans = pm.loc[mi, "answer_token"].astype(np.int64).to_numpy()
    y = np.asarray(t["y_hat"], np.int64)
    res = {}
    groups = {"aan_all": task == "agree_a_an", "aan_pred_an": (task == "agree_a_an") & (y == 459),
              "aan_pred_a": (task == "agree_a_an") & (y == 264), "other_tasks": task != "agree_a_an"}
    res["token_check"] = {"an_id_459_is_answer": int(((task == "agree_a_an") & (ans == 459)).sum()),
                          "a_id_264_is_answer": int(((task == "agree_a_an") & (ans == 264)).sum())}
    idx, s, h = t["idx"], t["s"], t["h"]
    n, L, k = idx.shape
    for g, m in groups.items():
        if not m.any():
            continue
        r = {"n": int(m.sum())}
        # each a/an neuron id in every layer: share of prompts with it in that layer's stored top-k and top-5
        for Lr, i in AAN:
            per = {}
            for l in range(L):
                hit = idx[m, l, :] == i
                inK, in5 = hit.any(1).mean(), hit[:, :5].any(1).mean()
                if inK > 0.2 or l == Lr:
                    per[l] = [round(float(in5), 3), round(float(inK), 3)]
            r[f"L{Lr}:{i}"] = per
        # leading neurons per layer 0-16 by top-5 frequency, with mean signed s and h when in top-5
        lead = {}
        for l in range(17):
            top5 = idx[m, l, :5].ravel()
            u, c = np.unique(top5, return_counts=True)
            o = np.argsort(-c)[:4]
            lead[l] = [[int(u[j]), round(float(c[j] / m.sum()), 3)] for j in o]
        r["lead_top5"] = lead
        res[g] = r
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({g: {kk: v for kk, v in r.items() if kk != "lead_top5"} for g, r in res.items() if g != "token_check"}, indent=1))
    print(res["token_check"])
    for g in ("aan_pred_an", "aan_pred_a", "other_tasks"):
        if g in res:
            print(g, {l: v for l, v in res[g]["lead_top5"].items() if int(l) in (2, 5, 8, 12, 13, 14)})


if __name__ == "__main__":
    main()
