"""Gate 1: does the attribution recover the published layer-18 arithmetic neurons?

For each task (addition, hours, months, weekdays; default templates, ``--n`` prompts each, fixed seed):

1. **Direct score.** Goodfire's original score at layer ``--layer``: |h * d log p(top-1) / dh| at the final position,
   for every neuron, by plain autograd (the forward pre-hook on the layer's down_proj detaches h and makes it the
   gradient leaf, so the backward pass only runs over the layers above). Rank neurons by the mean over prompts and
   count how many of the task's published neurons are in the top 40 (and top |set|). Also on the prompts the model
   gets right.
2. **Harvest path.** The same prompts through ``attribution-wide`` (k=50, the stage used for the real harvest), then
   the same ranking from the stored top-50 |s| (zero outside the top 50), so a bug in the harvest code shows up here
   even if the direct score is fine. Also: agreement of the two scores and the share of layer |s| the top 50 keep.

Writes ``<out>/gate1.json`` and ``<out>/gate1.md``; the harvest goes to ``<out>/harvest`` (Parquet).

  python scripts/gates/gate1_arith.py --model /path/to/snapshot --out outputs/gate1 --device cuda
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from neuron_communities.cli import main as nc_main                                  # noqa: E402
from neuron_communities.harvest.wide_blocks import load_wide                         # noqa: E402
from neuron_communities.slices import arithmetic as ar                               # noqa: E402
from neuron_communities.slices.prompts import write_prompt_world                     # noqa: E402
from neuron_communities.tail import load_model                                       # noqa: E402


def direct_scores(model, tok_ids: list[list[int]], layer: int, bs: int, dev: str):
    """(n, d_mlp) signed s at ``layer`` and the final position, plus top-1 ids and log-probs."""
    S, Y, LP = [], [], []
    mlp = model.model.layers[layer].mlp
    for i in range(0, len(tok_ids), bs):
        chunk = tok_ids[i: i + bs]
        L = max(len(t) for t in chunk)
        ids = torch.zeros(len(chunk), L, dtype=torch.long); att = torch.zeros_like(ids)
        for j, t in enumerate(chunk):
            ids[j, : len(t)] = torch.tensor(t); att[j, : len(t)] = 1
        ids, att = ids.to(dev), att.to(dev)
        tpos = att.sum(1) - 1
        cap = {}

        def pre(_m, ins):
            h = ins[0].detach().requires_grad_(True)
            cap["h"] = h
            return (h,)
        hd = mlp.down_proj.register_forward_pre_hook(pre)
        try:
            with torch.enable_grad():
                out = model.model(input_ids=ids, attention_mask=att)
                rows = torch.arange(len(chunk), device=dev)
                logits = model.lm_head(out.last_hidden_state[rows, tpos]).float()
                lp, y = torch.log_softmax(logits, -1).max(-1)
                (g,) = torch.autograd.grad(lp.sum(), cap["h"])
        finally:
            hd.remove()
        s = (cap["h"].detach().float() * g.float())[rows, tpos]
        S.append(s.cpu()); Y.append(y.cpu()); LP.append(lp.detach().cpu())
    return torch.cat(S).numpy(), torch.cat(Y).numpy(), torch.cat(LP).numpy()


def recovery(mean_abs: np.ndarray, target: list[int], top: int) -> dict:
    order = np.argsort(-mean_abs, kind="stable")
    rank = np.empty_like(order); rank[order] = np.arange(len(order))
    tset = set(target)
    return {"top40": int(len(tset & set(order[:top].tolist()))), "top_set_size": int(len(tset & set(order[: len(target)].tolist()))),
            "ranks": {int(n): int(rank[n]) for n in target}, "top40_ids": order[:top].tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=ar.LAYER)
    ap.add_argument("--top", type=int, default=40)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--tasks", default="addition,hours,months,weekdays")
    ap.add_argument("--neurons-json", type=Path, default=None, help="override the published sets (tests)")
    args = ap.parse_args()
    t0 = time.monotonic()
    args.out.mkdir(parents=True, exist_ok=True)
    tasks = args.tasks.split(",")
    neurons = json.loads(args.neurons_json.read_text()) if args.neurons_json else ar.NEURONS

    tok = AutoTokenizer.from_pretrained(args.model)
    recs = [r for t in tasks for r in ar.make_prompts(t, args.n, seed=args.seed)]
    world = write_prompt_world(recs, tok, args.out / "prompts", provenance={"generator": "slices.arithmetic.make_prompts",
                                                                           "n": args.n, "seed": args.seed, "templates": "default"})
    ids = [tok(r["text"], add_special_tokens=True)["input_ids"] for r in recs]
    gold = np.array([ar.first_answer_token(tok, r["text"], r["answer"]) for r in recs])

    model = load_model(args.model, args.device)
    s, y, lp = direct_scores(model, ids, args.layer, args.bs, args.device)
    t_direct = time.monotonic() - t0
    del model
    if args.device.startswith("cuda"):
        torch.cuda.empty_cache()

    # the harvest path, in-process, on the same prompts
    nc_main(["attribution-wide", "--manifest-dir", str(world), "--out-dir", str(args.out / "harvest"), "--shard", "0",
             "--shards", "1", "--split", "gate1", "--model", args.model, "--device", args.device])
    w = load_wide(args.out / "harvest", columns=["manifest_index", "y_hat", "idx", "s", "s_abs_sum"])
    d_mlp = s.shape[1]
    s_h = np.zeros((len(recs), d_mlp), np.float32)
    np.put_along_axis(s_h, w["idx"][:, args.layer].astype(np.int64), np.abs(w["s"][:, args.layer]), axis=1)
    keep_frac = np.abs(w["s"][:, args.layer]).sum(1) / w["s_abs_sum"][:, args.layer]

    task_arr = np.array([r["task"] for r in recs])
    res = {"model": args.model, "layer": args.layer, "n_per_task": args.n, "seed": args.seed, "top": args.top,
           "reference_top40": ar.REFERENCE_TOP40, "tasks": {}}
    for t in tasks:
        m = task_arr == t
        correct = m & (y == gold) & (gold >= 0)
        r = {"n": int(m.sum()), "accuracy_top1": float((y[m] == gold[m]).mean()), "n_gold_ambiguous": int((gold[m] < 0).sum()),
             "set_size": len(neurons[t]),
             "direct": recovery(np.abs(s[m]).mean(0), neurons[t], args.top),
             "direct_correct_only": recovery(np.abs(s[correct]).mean(0), neurons[t], args.top) if correct.sum() else None,
             "harvest_top50": recovery(s_h[m].mean(0), neurons[t], args.top),
             "harvest_vs_direct_top1_agree": float((w["y_hat"][m] == y[m]).mean()),
             "harvest_vs_direct_top40_overlap": None,
             "top50_share_of_layer_abs_s": {"mean": float(keep_frac[m].mean()), "p10": float(np.quantile(keep_frac[m], 0.1))},
             "signed_mean_s_of_set": float(s[m][:, neurons[t]].mean())}
        r["harvest_vs_direct_top40_overlap"] = len(set(r["direct"]["top40_ids"]) & set(r["harvest_top50"]["top40_ids"])) / args.top
        res["tasks"][t] = r
    res["seconds"] = time.monotonic() - t0
    res["seconds_direct"] = t_direct
    (args.out / "gate1.json").write_text(json.dumps(res, indent=1))

    lines = [f"# Gate 1: layer-{args.layer} arithmetic neuron recovery (top {args.top} by mean |s|, {args.n} prompts/task)", "",
             "| task | set | reference | direct | direct, correct only | harvest top-50 | accuracy | top-50 share of |s| |",
             "|---|---|---|---|---|---|---|---|"]
    for t, r in res["tasks"].items():
        dc = r["direct_correct_only"]["top40"] if r["direct_correct_only"] else "-"
        lines.append(f"| {t} | {r['set_size']} | {ar.REFERENCE_TOP40.get(t, '-')} | {r['direct']['top40']} | {dc} | "
                     f"{r['harvest_top50']['top40']} | {r['accuracy_top1']:.2f} | {r['top50_share_of_layer_abs_s']['mean']:.2f} |")
    (args.out / "gate1.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)
    print(f"gate1 done in {res['seconds'] / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
