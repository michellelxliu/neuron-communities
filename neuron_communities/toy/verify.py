"""Zero-ablation verification of a trained toy model on held-out (task, x) prompts.

* **necessity**: zero one primitive's full neuron set (final position) -> accuracy per task. Pass: every task that uses
  the set falls to near chance (<= ``--near-chance``, default 5%; chance is 1/n_symbols) and every other task stays >= 95%.
* **sufficiency**: per task, zero every neuron outside the task's blocks in both layers -> pass: accuracy >= 90%.
* **locality**: zero all free neurons in both layers -> pass: accuracy changes by < 2 points on every task.

Tables use fresh prompts (unseen random distractor contexts) of the trained (task, x) pairs. Also clean accuracy on training
prompts and on the compositional held-out pairs (never trained; solvable only through the shared blocks). Writes ``verification.json`` and ``verification.md``.

  python -m neuron_communities.toy.verify --run runs/toy/base
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import LlamaForCausalLM

from neuron_communities.toy.model import final_logits
from neuron_communities.toy.tasks import ToyConfig, ToyWorld


@torch.no_grad()
def accuracy_by_task(model, world: ToyWorld, data: dict, zero=None, bs: int = 2048) -> np.ndarray:
    nT = len(world.cfg.tasks)
    correct = np.zeros(nT); total = np.zeros(nT)
    for i in range(0, len(data["y"]), bs):
        tok = torch.as_tensor(data["tokens"][i: i + bs])
        task = data["task"][i: i + bs]
        z = None if zero is None else {L: zf(task) for L, zf in zero.items()}
        pred = final_logits(model, tok, zero=z).argmax(-1).numpy()
        ok = pred == data["y"][i: i + bs]
        for k in range(nT):
            correct[k] += ok[task == k].sum(); total[k] += (task == k).sum()
    return correct / np.maximum(total, 1)


def balanced(world: ToyWorld, split: str, n_per_task: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    nT = len(world.cfg.tasks)
    tasks = np.repeat(np.arange(nT), n_per_task)
    return world.sample(len(tasks), rng, split=split, tasks=tasks)


def verify(run: Path, n_per_task: int = 2000, seed: int = 123, near_chance: float = 0.05) -> dict:
    cfg = ToyConfig.from_json((run / "config.json").read_text())
    world = ToyWorld(cfg)
    model = LlamaForCausalLM.from_pretrained(run / "model").eval()
    D, nT = cfg.d_mlp, len(cfg.tasks)
    names = [f"T{k + 1}={a}∘{s}" for k, (s, a) in enumerate(cfg.tasks)]
    held = balanced(world, "train", n_per_task, seed)        # fresh prompts: unseen distractor contexts of the trained pairs
    train = balanced(world, "train", n_per_task, seed + 1)
    comp_tasks = [k for k in range(nT) if len(world.heldout_pairs[k])]
    comp = world.sample(n_per_task * len(comp_tasks), np.random.default_rng(seed + 2), split="heldout",
                        tasks=np.repeat(comp_tasks, n_per_task)) if comp_tasks else None
    const = lambda m: (lambda task: m.expand(len(task), D))

    res = {"tasks": names, "chance": 1 / cfg.n_symbols, "n_per_task": n_per_task,
           "clean_train": accuracy_by_task(model, world, train).tolist(),
           "clean_heldout": accuracy_by_task(model, world, held).tolist(),
           "compositional_heldout_pairs": {"n_pairs": [len(v) for v in world.heldout_pairs],
                                           "acc": accuracy_by_task(model, world, comp).tolist() if comp else None}}
    clean = np.array(res["clean_heldout"])

    def ablate(L, ids):
        m = torch.zeros(D); m[ids] = 1
        return accuracy_by_task(model, world, held, zero={L: const(m)})

    # necessity on each primitive's full set (the specified pass rule), plus its private part and each shared region
    nec, nec_pass = {}, True
    for p, (L, ids) in world.blocks.items():
        acc = ablate(L, ids)
        uses = world.M[:, world.primitives.index(p)].astype(bool)
        ok = bool((acc[uses] <= near_chance).all() and (acc[~uses] >= 0.95).all())
        nec[p] = {"acc": acc.tolist(), "uses": uses.tolist(), "pass": ok, "set": "full", "n": len(ids)}
        nec_pass &= ok
    for p, (L, ids) in world.private.items():
        if len(ids) < len(world.blocks[p][1]):
            uses = world.M[:, world.primitives.index(p)].astype(bool)
            acc = ablate(L, ids)
            nec[p + " private"] = {"acc": acc.tolist(), "uses": uses.tolist(), "set": "private", "n": len(ids),
                                   "pass": bool((acc[uses] <= near_chance).all() and (acc[~uses] >= 0.95).all())}
    for name, (L, ids) in world.shared.items():
        if ids:
            a_, b_ = name.split("&")
            uses = (world.M[:, world.primitives.index(a_)] | world.M[:, world.primitives.index(b_)]).astype(bool)
            nec[name] = {"acc": ablate(L, ids).tolist(), "uses": uses.tolist(), "set": "shared", "n": len(ids), "pass": None}
    res["necessity"] = nec

    keep = {}
    for L in (0, 1):
        um = torch.ones(nT, D)
        for k in range(nT):
            um[k, world.task_neurons(k)[L]] = 0
        keep[L] = (lambda task, um=um: um[torch.as_tensor(task)])
    suff = accuracy_by_task(model, world, held, zero=keep)
    res["sufficiency"] = {"acc": suff.tolist(), "pass": bool((suff >= 0.90).all())}

    fz = {}
    for L in (0, 1):
        m = torch.zeros(D); m[world.free[L]] = 1
        fz[L] = const(m)
    loc = accuracy_by_task(model, world, held, zero=fz)
    res["locality"] = {"acc": loc.tolist(), "delta_points": ((loc - clean) * 100).tolist(),
                       "pass": bool((np.abs(loc - clean) * 100 < 2).all())}
    res["pass"] = {"necessity": nec_pass, "sufficiency": res["sufficiency"]["pass"], "locality": res["locality"]["pass"]}
    (run / "verification.json").write_text(json.dumps(res, indent=1))
    (run / "verification.md").write_text(render(res, world))
    return res


def render(res: dict, world: ToyWorld) -> str:
    pct = lambda v: f"{100 * v:.1f}"
    hdr = "| | " + " | ".join(res["tasks"]) + " |\n|---|" + "---|" * len(res["tasks"])
    lines = [f"Fresh prompts (unseen distractor contexts), {res['n_per_task']} per task; chance {pct(res['chance'])}%. Accuracy in %.", "",
             "**Clean**", "", hdr,
             "| train | " + " | ".join(pct(v) for v in res["clean_train"]) + " |",
             "| fresh prompts | " + " | ".join(pct(v) for v in res["clean_heldout"]) + " |",
             "| held-out pairs (n pairs) | " + " | ".join(
                 (f"{pct(a)} ({n})" if n else "–") for a, n in zip(res["compositional_heldout_pairs"]["acc"] or [0] * len(res["tasks"]),
                                                             res["compositional_heldout_pairs"]["n_pairs"])) + " |", "",
             f"**Necessity** (zero a neuron set; * = task uses it; pass rule on full sets) — {'PASS' if res['pass']['necessity'] else 'FAIL'}", "", hdr]
    for p, r in res["necessity"].items():
        cells = [pct(a) + ("*" if u else "") for a, u in zip(r["acc"], r["uses"])]
        tag = "" if r["pass"] is None else (" ok" if r["pass"] else " FAIL")
        lines.append(f"| {p} ({r['n']} n){tag} | " + " | ".join(cells) + " |")
    lines += ["", f"**Sufficiency** (keep only the task's blocks) — {'PASS' if res['pass']['sufficiency'] else 'FAIL'}", "", hdr,
              "| accuracy | " + " | ".join(pct(v) for v in res["sufficiency"]["acc"]) + " |", "",
              f"**Locality** (zero all free neurons) — {'PASS' if res['pass']['locality'] else 'FAIL'}", "", hdr,
              "| accuracy | " + " | ".join(pct(v) for v in res["locality"]["acc"]) + " |",
              "| change (points) | " + " | ".join(f"{v:+.1f}" for v in res["locality"]["delta_points"]) + " |"]
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--n-per-task", type=int, default=2000)
    ap.add_argument("--near-chance", type=float, default=0.05)
    a = ap.parse_args()
    verify(a.run, a.n_per_task, near_chance=a.near_chance)
    print((a.run / "verification.md").read_text())


if __name__ == "__main__":
    main()
