"""Is the planted structure visible in the raw co-attribution graph of a toy model?

1. Harvest ``--n`` balanced fresh prompts with ``attribution-wide`` (all d_mlp neurons per layer, float32, objective
   ``logp`` as for Llama or ``logit_diff``).
2. Per prompt and layer take the top-``k`` neurons by |s| (k = 5, Goodfire's setting, and the extra ``--ks``); the
   co-attribution graph has the 2 x d_mlp neurons as nodes and, as edge weight, the number of prompts whose top-k
   sets contain both (raw counts), plus a z-score against independence, z = (c - f_i f_j / n) / sqrt(f_i f_j / n).
3. Report, against the planted ground truth:
   * **attribution precision**: share of each prompt's top-k (and of its total |s|) inside its task's circuit;
   * **modularity** of the planted partition (each primitive's private neurons, each shared region, free neurons),
     next to the mean modularity of 20 random relabelings with the same group sizes;
   * **within- vs between-circuit density**: mean edge weight over pairs that share a task circuit vs pairs of circuit
     neurons that share none;
   * **shared neurons**: for each neuron shared by primitives P and Q, its mean edge weight to P's side (the circuits
     of P's tasks, minus the shared region) and to Q's side, vs the between-circuit density; "connects to both" when
     both exceed it;
   * **saturation**: p(top-1) and the attributed objective's value.

Writes ``<run>/diagnostics_<objective>.json`` / ``.md``; the harvest goes to ``<run>/harvest_<objective>``.

  python -m neuron_communities.toy.diagnostics --run runs/toy/rho0 --objective logp
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import igraph as ig
import numpy as np

from neuron_communities import config
from neuron_communities.cli import main as nc_main
from neuron_communities.harvest.wide_blocks import load_wide
from neuron_communities.slices.prompts import write_prompt_world
from neuron_communities.toy.tasks import ToyConfig, ToyWorld
from neuron_communities.toy.verify import balanced


def harvest(run: Path, world: ToyWorld, n: int, objective: str, seed: int = 7) -> Path:
    config.configure(2, world.cfg.d_mlp)                   # toy geometry (== NEURON_COMMUNITIES_GEOMETRY=2x<d_mlp>)
    H = run / f"harvest_{objective}"
    if (H / "parts").exists() and any((H / "parts").glob("*.parquet")):
        return H
    nT = len(world.cfg.tasks)
    data = balanced(world, "train", max(1, n // nT), seed)
    recs = [{"prompt_id": f"{world.cfg.name}:{i}", "slice": world.cfg.name, "task": f"T{int(k) + 1}", "template_id": "toy",
             "split": "diagnostics", "text": " ".join(map(str, tok)), "token_ids": tok.tolist(), "x": int(x), "answer": int(y)}
            for i, (tok, k, x, y) in enumerate(zip(data["tokens"], data["task"], data["x"], data["y"]))]
    M = write_prompt_world(recs, None, run / "prompts_diag", provenance={"generator": "toy.verify.balanced", "seed": seed},
                           allow_duplicates=True)
    nc_main(["attribution-wide", "--manifest-dir", str(M), "--out-dir", str(H), "--shard", "0", "--shards", "1",
             "--k", str(world.cfg.d_mlp), "--block-size", "256", "--part-blocks", "8", "--dtype", "float32",
             "--objective", objective, "--model", str(run / "model"), "--device", "cpu", "--split", "diagnostics"])
    return H


def graph_metrics(X: np.ndarray, labels: np.ndarray, world: ToyWorld, rng: np.random.Generator) -> dict:
    """X: (n_prompts, 2*d_mlp) 0/1 top-k membership. labels: group id per node (planted partition)."""
    D = world.cfg.d_mlp
    n = len(X)
    C = (X.T.astype(np.float64) @ X).astype(np.float64)
    f = np.diag(C).copy()
    np.fill_diagonal(C, 0)
    E = np.outer(f, f) / n
    Z = np.where(E > 0, (C - E) / np.sqrt(np.maximum(E, 1e-12)), 0.0)
    np.fill_diagonal(Z, 0)
    active = f > 0
    out = {"active_nodes": int(active.sum())}

    def modularity(W, lab):
        idx = np.flatnonzero(active)
        Ws = np.clip(W[np.ix_(idx, idx)], 0, None)
        g = ig.Graph.Weighted_Adjacency(Ws.tolist(), mode="undirected", attr="weight", loops=False)
        return g.modularity(lab[idx].tolist(), weights="weight") if g.ecount() else float("nan")

    for name, W in (("raw", C), ("z_pos", Z)):
        q = modularity(W, labels)
        qr = [modularity(W, rng.permutation(labels)) for _ in range(20)]
        out[f"modularity_{name}"] = {"planted": q, "random_mean": float(np.mean(qr)), "random_sd": float(np.std(qr))}

    # circuits as global node ids (layer L neuron i -> L*D + i)
    circ = [set(i for L, ids in world.task_neurons(k).items() for i in (L * D + np.asarray(ids)).tolist())
            for k in range(len(world.cfg.tasks))]
    allc = sorted(set().union(*circ))
    co = np.zeros((2 * D, 2 * D), bool)
    for c in circ:
        cl = np.array(sorted(c)); co[np.ix_(cl, cl)] = True
    np.fill_diagonal(co, False)
    incirc = np.zeros(2 * D, bool); incirc[allc] = True
    between = incirc[:, None] & incirc[None, :] & ~co
    np.fill_diagonal(between, False)
    for name, W in (("raw", C), ("z", Z)):
        w_in = float(W[co].mean()); w_bt = float(W[between].mean()) if between.any() else float("nan")
        out[f"density_{name}"] = {"within": w_in, "between": w_bt, "ratio": w_in / w_bt if w_bt and w_bt > 0 else float("inf")}

    # shared neurons
    sh = []
    for key, (L, ids) in world.shared.items():
        if not ids:
            continue
        P, Q = key.split("&")
        shared_g = set((L * D + np.asarray(ids)).tolist())
        tP = [k for k, t in enumerate(world.cfg.tasks) if P in t]; tQ = [k for k, t in enumerate(world.cfg.tasks) if Q in t]
        sideP = set().union(*[circ[k] for k in tP]) - shared_g; sideQ = set().union(*[circ[k] for k in tQ]) - shared_g
        onlyP, onlyQ = np.array(sorted(sideP - sideQ)), np.array(sorted(sideQ - sideP))
        base = out["density_raw"]["between"]
        for g in sorted(shared_g):
            wp = float(C[g, onlyP].mean()) if len(onlyP) else float("nan")
            wq = float(C[g, onlyQ].mean()) if len(onlyQ) else float("nan")
            sh.append({"region": key, "node": int(g), "w_P_side": wp, "w_Q_side": wq, "between_density": base,
                       "both": bool(wp > base and wq > base), "freq": int(f[g])})
    if sh:
        out["shared"] = {"n": len(sh), "frac_connect_both": float(np.mean([s["both"] for s in sh])),
                         "frac_active": float(np.mean([s["freq"] > 0 for s in sh])),
                         "median_balance": float(np.nanmedian([min(s["w_P_side"], s["w_Q_side"]) / max(s["w_P_side"], s["w_Q_side"], 1e-12)
                                                              for s in sh])),
                         "per_region": {r: {"frac_connect_both": float(np.mean([s["both"] for s in sh if s["region"] == r])),
                                            "mean_w_P_side": float(np.nanmean([s["w_P_side"] for s in sh if s["region"] == r])),
                                            "mean_w_Q_side": float(np.nanmean([s["w_Q_side"] for s in sh if s["region"] == r]))}
                                        for r in sorted({s["region"] for s in sh})}}
    return out


def diagnose(run: Path, objective: str = "logp", n: int = 20000, ks: tuple = (5, 20), seed: int = 0) -> dict:
    cfg = ToyConfig.from_json((run / "config.json").read_text())
    world = ToyWorld(cfg)
    D = cfg.d_mlp
    H = harvest(run, world, n, objective)
    d = load_wide(H, columns=["task", "p_y_hat", "objective_value", "idx", "s", "s_abs_sum"])
    task = np.array([int(t[1:]) - 1 for t in d["task"]])
    n_p = len(task)
    # planted partition labels per global node: private(P) per primitive, shared regions, free per layer
    labels = np.full(2 * D, -1)
    gid = 0
    for p, (L, ids) in world.private.items():
        labels[L * D + np.asarray(ids)] = gid; gid += 1
    for key, (L, ids) in world.shared.items():
        if ids:
            labels[L * D + np.asarray(ids)] = gid; gid += 1
    for L in (0, 1):
        labels[L * D + np.asarray(world.free[L])] = gid; gid += 1
    assert (labels >= 0).all()
    circ_mask = np.zeros((len(cfg.tasks), 2, D), bool)
    for k in range(len(cfg.tasks)):
        for L, ids in world.task_neurons(k).items():
            circ_mask[k, L, ids] = True
    absS = np.zeros((n_p, 2, D), np.float32)
    np.put_along_axis(absS, d["idx"].astype(np.int64), np.abs(d["s"]), axis=2)
    res = {"run": str(run), "config": cfg.name, "overlap_rho": cfg.overlap_rho, "objective": objective, "n_prompts": n_p,
           "saturation": {"p_top1_median": float(np.median(d["p_y_hat"])), "p_top1_p10": float(np.quantile(d["p_y_hat"], 0.1)),
                          "objective_median": float(np.median(d["objective_value"])),
                          "sum_abs_s_median_by_layer": np.median(d["s_abs_sum"], 0).tolist()},
           "mass_in_circuit_by_layer": [float((absS[:, L] * circ_mask[task, L]).sum(1).mean() / absS[:, L].sum(1).mean()) for L in (0, 1)],
           "by_k": {}}
    rng = np.random.default_rng(seed)
    for k in ks:
        top = np.argsort(-absS, axis=2)[:, :, :k]                                    # (n, 2, k)
        X = np.zeros((n_p, 2 * D), np.uint8)
        for L in (0, 1):
            np.put_along_axis(X[:, L * D:(L + 1) * D], top[:, L], 1, axis=1)
        prec = [float(np.take_along_axis(circ_mask[task, L], top[:, L], axis=1).mean()) for L in (0, 1)]
        res["by_k"][str(k)] = {"topk_precision_by_layer": prec, **graph_metrics(X, labels, world, rng)}
    (run / f"diagnostics_{objective}.json").write_text(json.dumps(res, indent=1))
    (run / f"diagnostics_{objective}.md").write_text(render(res))
    return res


def render(r: dict) -> str:
    L = [f"**{r['config']}** (rho = {r['overlap_rho']}), objective `{r['objective']}`, {r['n_prompts']} prompts. "
         f"p(top-1) median {r['saturation']['p_top1_median']:.4f} (p10 {r['saturation']['p_top1_p10']:.4f}); "
         f"share of |s| inside the task circuit: L0 {r['mass_in_circuit_by_layer'][0]:.2f}, L1 {r['mass_in_circuit_by_layer'][1]:.2f}.", "",
         "| top-k | precision L0 / L1 | modularity raw (random) | modularity z+ (random) | within/between raw | within/between z | shared: connect both |",
         "|---|---|---|---|---|---|---|"]
    for k, m in r["by_k"].items():
        sh = f"{m['shared']['frac_connect_both']:.2f} of {m['shared']['n']}" if "shared" in m else "–"
        L.append(f"| {k} | {m['topk_precision_by_layer'][0]:.2f} / {m['topk_precision_by_layer'][1]:.2f} | "
                 f"{m['modularity_raw']['planted']:.3f} ({m['modularity_raw']['random_mean']:.3f}) | "
                 f"{m['modularity_z_pos']['planted']:.3f} ({m['modularity_z_pos']['random_mean']:.3f}) | "
                 f"{m['density_raw']['ratio']:.1f} | {m['density_z']['ratio']:.1f} | {sh} |")
    return "\n".join(L) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--objective", choices=("logp", "logit_diff"), default="logp")
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--ks", default="5,20")
    a = ap.parse_args()
    diagnose(a.run, a.objective, a.n, tuple(int(x) for x in a.ks.split(",")))
    print((a.run / f"diagnostics_{a.objective}.md").read_text())


if __name__ == "__main__":
    main()
