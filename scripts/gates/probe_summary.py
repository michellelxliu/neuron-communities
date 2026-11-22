"""Summarise a wide harvest run as a memory and throughput probe (and a gate-3 preview).

Reads ``<harvest>/parts/*.parquet``: per part prompts, tokens, seconds and peak GPU memory, the crosscheck results,
the prompt-length distribution, and per layer the share of total |s| kept by the top k. Extrapolates the time of a
run of ``--target-prompts`` prompts with the same length distribution on ``--slices`` devices like this one.

  python scripts/gates/probe_summary.py --harvest outputs/probe/harvest --target-prompts 2100000 --slices 16
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
    ap.add_argument("--target-prompts", type=int, default=2_100_000)
    ap.add_argument("--slices", type=int, default=16)
    ap.add_argument("--skip-first", type=int, default=1, help="parts left out of the rate (warm-up)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    metas = [part_meta(p) for p in part_files(args.harvest)]
    w = load_wide(args.harvest, columns=["position", "s", "s_abs_sum"])
    lens = w["position"] + 1
    share = np.abs(w["s"]).sum(-1) / np.maximum(w["s_abs_sum"], 1e-30)            # (n, L)
    timed = metas[args.skip_first:] or metas
    secs = sum(m["seconds"] for m in timed); n = sum(m["prompts"] for m in timed); toks = sum(m["tokens"] for m in timed)
    rate = n / secs * 3600
    cc = [c for m in metas for b in m["blocks"] for c in b["crosscheck"]]
    rel = [c for m in metas for b in m["blocks"] for c in b["crosscheck_rel_err"]]
    out = {
        "parts": [{k: m[k] for k in ("part", "prompts", "tokens", "seconds", "peak_mem_gib", "crosscheck_max_tokens")} for m in metas],
        "prompt_tokens": {"mean": float(lens.mean()), "p50": float(np.median(lens)), "p90": float(np.quantile(lens, 0.9)),
                          "max": int(lens.max())},
        "prompts_per_hour_per_device": rate, "tokens_per_second_per_device": toks / secs,
        "peak_mem_gib_max": max((m["peak_mem_gib"] or 0) for m in metas),
        "crosscheck": {"n": len(cc), "topk_overlap_mean": float(np.mean(cc)) if cc else None,
                       "grad_rel_err_max": float(np.max(rel)) if rel else None,
                       "skipped_subbatches": sum(b.get("crosscheck_skipped", 0) for m in metas for b in m["blocks"])},
        "nonfinite_h": sum(b["nonfinite_h"] for m in metas for b in m["blocks"]),
        "topk_share_of_abs_s_by_layer": {"mean": share.mean(0).round(4).tolist(), "p10": np.quantile(share, 0.1, axis=0).round(4).tolist()},
        "extrapolation": {"target_prompts": args.target_prompts, "devices": args.slices,
                          "device_hours": args.target_prompts / rate, "wall_hours": args.target_prompts / rate / args.slices},
    }
    s = json.dumps(out, indent=1)
    print(s)
    if args.out:
        args.out.write_text(s)


if __name__ == "__main__":
    main()
