"""Build the arithmetic slice: two prompt worlds (discovery, heldout) + ground_truth.json.

  python scripts/slices/build_arithmetic.py --model SNAPSHOT --out WORK/arith

<out>/discovery, <out>/heldout: prompt worlds (``slices.prompts.write_prompt_world``) whose prompts.parquet carries
task, template_id, input/offset (prompt strings and causal values), answer, answer_full and answer_token (the first
token of the answer after the prompt, for accuracy and logit-difference analyses). <out>/ground_truth.json: N_add, the
per-task subsets and the period clusters. <out>/slice_summary.json: counts per split, task and template.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuron_communities.slices import arithmetic as ar                    # noqa: E402
from neuron_communities.slices.prompts import write_prompt_world          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="tokenizer (local snapshot path pins the revision)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0, help="seed of the value-pair partition")
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    sl = ar.build_slice(a.seed)
    summary = {}
    for split, recs in sl.items():
        for r in recs:
            r["answer_token"] = str(ar.first_answer_token(tok, r["text"], r["answer"]))
        assert all(int(r["answer_token"]) >= 0 for r in recs)
        write_prompt_world(recs, tok, a.out / split, provenance={
            "generator": "neuron_communities.slices.arithmetic.build_slice", "seed": a.seed,
            "source": "github.com/goodfire-ai/arithmetic-wild @ d03024a, src/rephrased_templates.ipynb (Table 7)",
            "templates": {t: list(v[split]) for t, v in ar.SPLIT_TEMPLATES.items()}, "tokenizer": str(a.model)})
        summary[split] = {f"{t}:{tid}": n for (t, tid), n in sorted(Counter((r["task"], r["template_id"]) for r in recs).items())}
        summary[split]["total"] = len(recs)
    (a.out / "ground_truth.json").write_text(json.dumps(ar.ground_truth(), indent=1))
    (a.out / "templates.json").write_text(json.dumps({t: [{"template_id": tid, "template": tpl, "input_map": im,
        "offset_map": om, "cycles": cy} for tid, tpl, im, om, cy in rows] for t, rows in ar.TABLE7.items()}, indent=1))
    (a.out / "slice_summary.json").write_text(json.dumps(summary, indent=1))
    print(json.dumps({s: v["total"] for s, v in summary.items()}))


if __name__ == "__main__":
    main()
