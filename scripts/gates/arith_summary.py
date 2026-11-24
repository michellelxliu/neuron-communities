"""Arithmetic slice summary from a wide harvest: accuracy and layer-18 recovery of the published neuron sets.

Per split and task (and per template): accuracy (top-1 == the answer's first token); the gate-1 statistic on the
harvested scores (rank layer-18 neurons by mean |s| over prompts, zero outside each prompt's stored top-k; count the
task's published neurons in the top 40 and the top |set|), on all prompts and on the correct ones; and per period
cluster the share of its neurons in the addition top 40.

  python scripts/gates/arith_summary.py --harvest H --prompts P --truth ground_truth.json --out summary.json
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

D = 14336


def recovery(mean_abs: np.ndarray, truth: list[int], top: int) -> int:
    return int(len(set(np.argsort(-mean_abs)[:top].tolist()) & set(truth)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--harvest", type=Path, required=True)
    ap.add_argument("--prompts", type=Path, required=True, help="prompt world dir (prompts.parquet)")
    ap.add_argument("--truth", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=18)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", default=None, help="tokenizer: also accept the full-name answer's first token (' February' "
                                                   "for a prompt that writes 'Feb'), as the arithmetic-wild notebook scores")
    a = ap.parse_args()
    gt = json.loads(a.truth.read_text())
    t = load_wide(a.harvest, columns=["manifest_index", "y_hat", "idx", "s"])
    pm = pq.read_table(a.prompts / "prompts.parquet").to_pandas().set_index("manifest_index")
    mi = np.asarray(t["manifest_index"], np.int64)
    idx = t["idx"][:, a.layer, :].astype(np.int64)                          # (n, k), descending |s|
    s = np.abs(t["s"][:, a.layer, :].astype(np.float32))
    k = idx.shape[1]
    dense = np.zeros((len(mi), D), np.float32)
    np.put_along_axis(dense, idx, s, axis=1)
    task = pm.loc[mi, "task"].to_numpy(); tid = pm.loc[mi, "template_id"].to_numpy()
    y_hat = np.asarray(t["y_hat"], np.int64)
    strict = y_hat == pm.loc[mi, "answer_token"].astype(np.int64).to_numpy()
    correct = strict.copy()
    if a.model and "answer_full" in pm.columns:
        from transformers import AutoTokenizer
        from neuron_communities.slices.arithmetic import first_answer_token
        tok = AutoTokenizer.from_pretrained(a.model)
        full_tok = np.array([first_answer_token(tok, x, f) for x, f in zip(pm.loc[mi, "text"], pm.loc[mi, "answer_full"])])
        correct |= y_hat == full_tok
    res = {"n": int(len(mi)), "k_stored": k, "layer": a.layer, "tasks": {}}
    for tk in sorted(set(task)):
        m = task == tk
        truth = gt["per_task"][tk]
        r = {"n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4), "accuracy_strict": round(float(strict[m].mean()), 4),
             "top40": recovery(dense[m].mean(0), truth, 40), f"top{len(truth)}": recovery(dense[m].mean(0), truth, len(truth)),
             "set_size": len(truth)}
        mc = m & correct
        if mc.sum() >= 10:
            r["top40_correct_only"] = recovery(dense[mc].mean(0), truth, 40)
        r["templates"] = {}
        for tp in sorted(set(tid[m])):
            mt = m & (tid == tp)
            r["templates"][tp] = {"n": int(mt.sum()), "accuracy": round(float(correct[mt].mean()), 4),
                                  "accuracy_strict": round(float(strict[mt].mean()), 4),
                                  "top40": recovery(dense[mt].mean(0), truth, 40)}
        if tk == "addition":
            top40 = set(np.argsort(-dense[m].mean(0))[:40].tolist())
            r["period_clusters_in_top40"] = {T: f"{len(top40 & set(v))}/{len(v)}" for T, v in gt["period_clusters"].items()}
        res["tasks"][tk] = r
    res["reference_top40"] = gt["reference_top40"]
    a.out.write_text(json.dumps(res, indent=1))
    print(json.dumps({tk: {kk: v[kk] for kk in ("n", "accuracy", "top40", "set_size")} for tk, v in res["tasks"].items()}, indent=1))


if __name__ == "__main__":
    main()
