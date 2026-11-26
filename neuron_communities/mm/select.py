"""Shared machinery for both methods: held-out cells and pairs, choice of K, stability across seeds, membership sets.

Loadings L (m neurons x K) are what both methods hand over: Method 1 gives beta (columns sum to 1), Method 2 gives
E[pi] (rows sum to 1). Membership (plan): neuron n belongs to sub-community k when its share L_nk / sum_k' L_nk' is at
least ``share`` (default 0.2) and L_nk is at least ``min_rel`` times the largest loading in column k (so a neuron that is
barely used anywhere is not a member just because its shares sum to 1). A neuron in two or more sub-communities is
shared.

Stability: fit with several seeds; the reference fit is the one with the highest training objective. Every other seed's
columns are matched to the reference's by the Hungarian method on cosine similarity. A reference column is stable when
a match with cosine >= ``min_cos`` occurs in at least ``min_frac`` of all seeds (the reference counts as one).
"""
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import linear_sum_assignment

SIZE_CAP = 128        # the paper's sparsity bound per community


# ---------- held-out sets ----------

def sample_cells(shape: tuple[int, int], frac: float = 0.1, *, max_cells: int = 1_000_000, seed: int = 0):
    """Uniform sample of matrix cells (zeros included) for held-out scoring of Method 1: (rows, cols)."""
    n, m = shape
    total = n * m
    k = int(min(max(frac * total, 1), max_cells, total))
    lin = np.random.default_rng(seed).choice(total, size=k, replace=False)
    return lin // m, lin % m


def sample_pairs(m: int, frac: float = 0.1, *, max_pairs: int = 200_000, seed: int = 0):
    """Uniform sample of node pairs a < b (edges and non-edges alike) for held-out scoring of Method 2."""
    total = m * (m - 1) // 2
    k = int(min(max(frac * total, 1), max_pairs, total))
    rng = np.random.default_rng(seed)
    got = np.zeros(0, dtype=np.int64)
    while len(got) < k:
        u = rng.integers(0, m, size=(2 * k, 2))
        lo, hi = u.min(1), u.max(1)
        got = np.unique(np.concatenate([got, lo[lo != hi] * m + hi[lo != hi]]))
    got = rng.permutation(got)[:k]
    return got // m, got % m


# ---------- choice of K ----------

def choose_k(scores: dict[int, list[float]], *, rel_tol: float = 1e-3) -> tuple[int, dict]:
    """Smallest K whose mean held-out log-likelihood is within max(one standard error of the best K, rel_tol x |best|)
    of the best mean (the one-standard-error rule). scores: K -> held-out log-likelihoods over seeds."""
    summ = {}
    for K, v in scores.items():
        v = np.asarray(v, dtype=float)
        se = float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0
        summ[K] = {"mean": float(v.mean()), "se": se, "n": len(v)}
    best = max(summ, key=lambda K: summ[K]["mean"])
    thr = summ[best]["mean"] - max(summ[best]["se"], rel_tol * abs(summ[best]["mean"]))
    chosen = min(K for K in summ if summ[K]["mean"] >= thr)
    return chosen, {"best": best, "threshold": thr, "by_k": summ}


# ---------- stability ----------

def _cos(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    A = A / np.maximum(np.linalg.norm(A, axis=0, keepdims=True), 1e-12)
    B = B / np.maximum(np.linalg.norm(B, axis=0, keepdims=True), 1e-12)
    return A.T @ B


def match(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian match of other's columns to ref's: (col of other for each ref col, or -1; cosine of the match)."""
    C = _cos(ref, other)
    ri, oi = linear_sum_assignment(-C)
    col = np.full(ref.shape[1], -1)
    cos = np.zeros(ref.shape[1])
    col[ri], cos[ri] = oi, C[ri, oi]
    return col, cos


def stable_fit(fit_fn: Callable[[int], tuple[np.ndarray, float, dict]], seeds, *, min_cos: float = 0.8,
               min_frac: float = 0.8) -> dict:
    """Run fit_fn(seed) -> (loadings m x K, training objective, fit) for every seed; return the reference fit, the
    recurrence of each of its columns, and which columns are stable."""
    runs = [(s, *fit_fn(s)) for s in seeds]
    ref_i = int(np.argmax([r[2] for r in runs]))
    ref = runs[ref_i][1]
    hits = np.zeros(ref.shape[1])
    cos_all = []
    for i, (s, L, obj, _) in enumerate(runs):
        if i == ref_i:
            hits += 1
            continue
        _, cos = match(ref, L)
        cos_all.append(cos)
        hits += cos >= min_cos
    recurrence = hits / len(runs)
    return {"ref_seed": runs[ref_i][0], "loadings": ref, "fit": runs[ref_i][3], "objective": runs[ref_i][2],
            "recurrence": recurrence, "stable": recurrence >= min_frac,
            "match_cos": np.array(cos_all) if cos_all else np.zeros((0, ref.shape[1]))}


# ---------- membership ----------

def membership_sets(L: np.ndarray, *, share: float = 0.2, min_rel: float = 0.05) -> list[np.ndarray]:
    """Member neuron indices of each column of L (m x K, nonnegative)."""
    L = np.asarray(L, dtype=float)
    sh = L / np.maximum(L.sum(1, keepdims=True), 1e-300)
    rel = L / np.maximum(L.max(0, keepdims=True), 1e-300)
    M = (sh >= share) & (rel >= min_rel)
    return [np.flatnonzero(M[:, k]) for k in range(L.shape[1])]


def summarize(sets: list[np.ndarray], m: int, keep: np.ndarray | None = None, *, size_cap: int = SIZE_CAP) -> dict:
    """Sub-communities (the kept columns with >= 2 members), shared neurons (in >= 2 of them), pairwise shared sets."""
    idx = [k for k in range(len(sets)) if (keep is None or keep[k]) and len(sets[k]) >= 2]
    count = np.zeros(m, dtype=int)
    for k in idx:
        count[sets[k]] += 1
    pairs = []
    for x in range(len(idx)):
        for y in range(x + 1, len(idx)):
            S = np.intersect1d(sets[idx[x]], sets[idx[y]])
            if len(S):
                pairs.append({"a": idx[x], "b": idx[y], "shared": S.tolist()})
    return {"components": idx, "sizes": [len(sets[k]) for k in idx],
            "over_cap": [k for k in idx if len(sets[k]) > size_cap],
            "shared": np.flatnonzero(count >= 2).tolist(), "unassigned": np.flatnonzero(count == 0).tolist(),
            "pairs": pairs}


def jaccard(a, b) -> float:
    a, b = set(np.asarray(a).tolist()), set(np.asarray(b).tolist())
    return len(a & b) / max(len(a | b), 1)
