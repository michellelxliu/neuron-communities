"""Build the exploratory slices: two prompt worlds (discovery, heldout) of ``slices.exploratory.build_slices``.

  python scripts/slices/build_exploratory.py --model SNAPSHOT --out WORK/explore

Each prompt gets ``answer_token`` and ``alt_token`` (-1 when there is no competing form). When the expected and the
competing continuation share leading tokens (" läuft" / " laufen" both start with " l"), the prompt is extended by the
shared tokens (``shared_prefix`` records them), so the target is the first token where the two forms differ. Prompts
whose answer would merge with the prompt's last token are dropped (counted in slice_summary.json).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuron_communities.slices import exploratory as ex                     # noqa: E402
from neuron_communities.slices.prompts import write_prompt_world           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    recs = ex.build_slices(a.seed)
    kept, dropped = [], Counter()
    for r in recs:
        p = tok(r["text"], add_special_tokens=True)["input_ids"]
        fa = tok(r["text"] + r["answer"], add_special_tokens=True)["input_ids"]
        if fa[: len(p)] != p or len(fa) <= len(p):
            dropped[r["task"]] += 1
            continue
        j, alt_tok = len(p), -1
        if r["answer_alt"]:
            fb = tok(r["text"] + r["answer_alt"], add_special_tokens=True)["input_ids"]
            if fb[: len(p)] != p:
                dropped[r["task"]] += 1
                continue
            while j < min(len(fa), len(fb)) and fa[j] == fb[j]:
                j += 1
            if j >= min(len(fa), len(fb)):
                dropped[r["task"]] += 1
                continue
            alt_tok = fb[j]
        r["shared_prefix"] = tok.decode(fa[len(p): j]) if j > len(p) else ""
        r["text"] = r["text"] + r["shared_prefix"]
        r["token_ids"] = fa[:j]
        r["answer_token"] = str(fa[j]); r["alt_token"] = str(alt_tok)
        kept.append(r)
    fields = sorted({k for r in kept for k in r})
    for r in kept:                                   # same columns for every record (features differ by task)
        for f in fields:
            r.setdefault(f, "")
    summary = {"dropped_answer_merges": dict(dropped)}
    for split in ("discovery", "heldout"):
        rs = [r for r in kept if r["split"] == split]
        write_prompt_world(rs, tok, a.out / split, provenance={"generator": "neuron_communities.slices.exploratory.build_slices",
                                                                "seed": a.seed, "tokenizer": str(a.model)})
        summary[split] = {f"{t}:{tid}": n for (t, tid), n in sorted(Counter((r["task"], r["template_id"]) for r in rs).items())}
        summary[split]["total"] = len(rs)
    (a.out / "slice_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({k: (v if k == "dropped_answer_merges" else v["total"]) for k, v in summary.items()}))


if __name__ == "__main__":
    main()
