"""Plant (possibly overlapping) task circuits in the toy model with task-specific masks, then fine-tune without them.

Phase 1, **masked** (``steps_mask``): every row's final-position MLP activations outside its task's circuit (the union
of its two primitives' neuron sets, in both layers) are zeroed, so task t can only use its own neurons; neurons shared
by two primitives are used by the tasks of both. Loss: cross-entropy of ``y`` at ``=``.

Phase 2, **fine-tune** (``steps_finetune``): masks removed; loss = cross-entropy + ``off_penalty`` x the share of the
final-position activation mass (sum |h|, both layers) that falls outside the task's circuit, which keeps the trained
circuits in place while the model runs unmasked.

SIIT (Gupta et al., arXiv:2407.14494) is not used: aligning two high-level variables to intersecting neuron sets makes
the interchange interventions conflict on the shared neurons. ``label_smoothing`` > 0 keeps softmaxes unsaturated.

  python -m neuron_communities.toy.train --out runs/toy/rho0 [--config cfg.json] [--rho 0.25] [--null]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from neuron_communities.harvest.wide import code_commit
from neuron_communities.toy.model import build_model, final_logits, intervene
from neuron_communities.toy.tasks import ToyConfig, ToyWorld


def off_circuit_masks(world: ToyWorld, device="cpu") -> dict[int, torch.Tensor]:
    """Per layer, a (n_tasks, d_mlp) mask: 1 where the neuron is outside the task's circuit."""
    nT, D = len(world.cfg.tasks), world.cfg.d_mlp
    off = {L: torch.ones(nT, D, device=device) for L in (0, 1)}
    for k in range(nT):
        for L, ids in world.task_neurons(k).items():
            off[L][k, ids] = 0
    return off


def train(cfg: ToyConfig, out: Path, device: str = "cpu", log_every: int = 500) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    world = ToyWorld(cfg)
    model = build_model(cfg, world).to(device)
    off = off_circuit_masks(world, device)
    rng = np.random.default_rng(cfg.seed + 1)
    T = lambda a: torch.as_tensor(a, device=device)
    log, t0 = [], time.monotonic()

    def run_phase(name: str, steps: int, lr: float, masked: bool):
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=cfg.weight_decay)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=max(steps, 1), pct_start=0.05)
        for step in range(steps):
            model.train()
            b = world.sample(cfg.batch, rng)
            tok, y, task = T(b["tokens"]), T(b["y"]), T(b["task"])
            cap: dict = {}
            zero = {L: off[L][task] for L in (0, 1)} if masked else None
            with intervene(model, zero=zero, capture=cap):
                logits = model.lm_head(model.model(input_ids=tok).last_hidden_state[:, -1])
            ce = F.cross_entropy(logits, y, label_smoothing=cfg.label_smoothing)
            a = {L: cap[L].abs() for L in (0, 1)}
            off_mass = sum((a[L] * off[L][task]).sum(-1) for L in (0, 1))
            share = (off_mass / (sum(a[L].sum(-1) for L in (0, 1)) + 1e-8)).mean()
            loss = ce + (0.0 if masked else cfg.off_penalty) * share
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            if step % log_every == 0 or step == steps - 1:
                acc = (logits.argmax(-1) == y).float().mean().item()
                rec = {"phase": name, "step": step, "ce": ce.item(), "off_share": share.item(), "acc": acc,
                       "min": (time.monotonic() - t0) / 60}
                log.append(rec)
                print(json.dumps(rec), flush=True)

    run_phase("mask", cfg.steps_mask, cfg.lr, masked=True)
    run_phase("finetune", cfg.steps_finetune, cfg.lr_finetune, masked=False)
    model.eval()
    model.save_pretrained(out / "model")
    (out / "config.json").write_text(cfg.to_json())
    (out / "ground_truth.json").write_text(json.dumps(world.ground_truth()))
    meta = {"code": code_commit(), "torch": torch.__version__, "device": device, "train_minutes": (time.monotonic() - t0) / 60,
            "method": "task-specific overlapping masks, then unmasked fine-tune with an off-circuit activation penalty", "log": log}
    (out / "train_meta.json").write_text(json.dumps(meta, indent=1))
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, default=None, help="ToyConfig JSON (default: the base configuration)")
    ap.add_argument("--rho", type=float, default=None, help="override overlap_rho")
    ap.add_argument("--null", action="store_true", help="train the matched null of the configuration")
    ap.add_argument("--steps-mask", type=int, default=None)
    ap.add_argument("--steps-finetune", type=int, default=None)
    ap.add_argument("--label-smoothing", type=float, default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=0)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    cfg = ToyConfig.from_json(a.config.read_text()) if a.config else ToyConfig()
    for k, v in (("overlap_rho", a.rho), ("steps_mask", a.steps_mask), ("steps_finetune", a.steps_finetune),
                 ("label_smoothing", a.label_smoothing), ("name", a.name)):
        if v is not None:
            setattr(cfg, k, v)
    if a.null:
        cfg = ToyConfig.null_of(cfg)
    train(cfg, a.out, a.device)


if __name__ == "__main__":
    main()
