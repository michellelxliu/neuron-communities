"""Method 2: assortative mixed-membership stochastic blockmodel (a-MMSB) on the parent community's co-occurrence graph.

Model (Airoldi et al. 2008, assortative form of Gopalan & Blei 2013): neuron a has memberships pi_a ~ Dirichlet(alpha)
over K blocks; block k has a within-block edge probability b_k ~ Beta(eta1, eta0). For each pair (a, b), a draws a
block z_ab ~ pi_a and b draws z_ba ~ pi_b; the edge y_ab ~ Bernoulli(b_k) if z_ab = z_ba = k, else Bernoulli(eps).
The assortative form fits K within-block rates instead of a K x K matrix: the right size for communities, and the cost
per pair is O(K). eps (the between-block rate) is fixed, by default 0.1 x the graph's edge density.

Fit: full-batch mean-field variational inference. Per pair, phi_ab (q of z_ab) and phi_ba alternate:
  phi_ab,k  propto  exp(E[log pi_ak] + phi_ba,k * (E[log p(y | b_k)] - log p(y | eps)))
then g_a = alpha + sum of phi over a's pairs (the Dirichlet parameters of q(pi_a)) and
lam_k = (eta1 + sum phi_ab,k phi_ba,k y_ab, eta0 + sum phi_ab,k phi_ba,k (1 - y_ab)).

Pairs: all m(m-1)/2 when that is at most ``max_pairs``; otherwise every edge plus a uniform sample of non-edges,
each sampled non-edge weighted by (number of non-edges) / (number sampled). Held-out pairs get weight 0.

Memberships: E[pi_a] = g_a / sum(g_a).
"""
from __future__ import annotations

import numpy as np
from scipy.special import digamma

EPS = 1e-12


def _lin(i, j, m):
    return np.asarray(i, dtype=np.int64) * m + np.asarray(j, dtype=np.int64)


def _norm_edges(edges, m):
    i, j = (np.asarray(a, dtype=np.int64) for a in edges)
    lo, hi = np.minimum(i, j), np.maximum(i, j)
    keep = lo != hi
    lin = np.unique(_lin(lo[keep], hi[keep], m))
    if len(lin) and (lin.min() < 0 or lin.max() >= m * m):
        raise ValueError("edge endpoint out of range")
    return lin


def build_pairs(m: int, edges, *, max_pairs: int = 2_000_000, seed: int = 0, heldout=None):
    """Pair list for the fit: a, b (a < b), y (bool), w (weight). Held-out pairs (row, col arrays) get weight 0."""
    elin = _norm_edges(edges, m)
    total = m * (m - 1) // 2
    if total <= max_pairs:
        a, b = np.triu_indices(m, 1)
        a, b = a.astype(np.int64), b.astype(np.int64)
        y = np.isin(_lin(a, b, m), elin)
        w = np.ones(len(a))
    else:
        rng = np.random.default_rng(seed)
        n_non = total - len(elin)
        k = max(max_pairs - len(elin), 1)
        got = np.zeros(0, dtype=np.int64)
        while len(got) < min(k, n_non):
            u = rng.integers(0, m, size=(2 * k, 2))
            lo, hi = u.min(1), u.max(1)
            lin = _lin(lo[lo != hi], hi[lo != hi], m)
            lin = lin[~np.isin(lin, elin)]
            got = np.unique(np.concatenate([got, lin]))
        got = rng.permutation(got)[:min(k, n_non)]
        lin = np.concatenate([elin, got])
        a, b = lin // m, lin % m
        y = np.concatenate([np.ones(len(elin), bool), np.zeros(len(got), bool)])
        w = np.concatenate([np.ones(len(elin)), np.full(len(got), n_non / max(len(got), 1))])
    if heldout is not None:
        hl = _norm_edges(heldout, m)
        w[np.isin(_lin(a, b, m), hl)] = 0.0
    return a, b, y, w


def _softmax(z):
    z = z - z.max(1, keepdims=True)
    np.exp(z, out=z)
    z /= z.sum(1, keepdims=True)
    return z


def fit_mmsb(m: int, edges, K: int, *, alpha: float | None = None, eta: tuple[float, float] = (1.0, 1.0),
             eps: float | None = None, heldout=None, max_pairs: int = 2_000_000, n_iter: int = 500,
             tol: float = 1e-5, inner: int = 2, seed: int = 0, init: str = "nmf") -> dict:
    """Fit a-MMSB to the graph on nodes 0..m-1 with edge list ``edges`` (two index arrays). Returns pi (m x K,
    E[pi]), g (Dirichlet parameters), b (E[b_k]), lam, eps, loglik (plug-in, training pairs), n_iter, converged.

    init="nmf" (default) starts phi from a KL-NMF of the adjacency matrix (rows = nodes; random start of its own, from
    ``seed``); init="random" draws phi from Dirichlet(1). Random starts often end in one block that absorbs most
    nodes."""
    a, b, y, w = build_pairs(m, edges, max_pairs=max_pairs, seed=seed, heldout=heldout)
    total = m * (m - 1) // 2
    n_edges = int(w[y].sum()) if len(y) else 0
    if eps is None:
        eps = max(0.1 * n_edges / max(total, 1), 1e-6)
    alpha = 1.0 / K if alpha is None else alpha
    eta1, eta0 = eta
    rng = np.random.default_rng(seed)
    P = len(a)
    if init == "nmf":
        from scipy import sparse
        from neuron_communities.mm.pf import fit_pf
        ea, eb = a[y & (w > 0)], b[y & (w > 0)]
        A = sparse.csr_matrix((np.ones(2 * len(ea)), (np.r_[ea, eb], np.r_[eb, ea])), shape=(m, m))
        H = fit_pf(A, K, seed=seed, n_iter=300)["theta"] + 1e-3
        H /= H.sum(1, keepdims=True)
        phi_ab, phi_ba = H[a].copy(), H[b].copy()
    elif init == "random":
        phi_ab = rng.dirichlet(np.ones(K), size=P)
        phi_ba = rng.dirichlet(np.ones(K), size=P)
    else:
        raise ValueError(init)

    def scatter(phi1, phi2):
        g = np.full((m, K), alpha)
        for k in range(K):
            g[:, k] += np.bincount(a, weights=w * phi1[:, k], minlength=m)
            g[:, k] += np.bincount(b, weights=w * phi2[:, k], minlength=m)
        return g

    g = scatter(phi_ab, phi_ba)
    pp = (phi_ab * phi_ba) * w[:, None]
    lam1 = eta1 + pp[y].sum(0)
    lam0 = eta0 + pp[~y].sum(0)
    log_eps, log_1meps = np.log(eps), np.log1p(-eps)
    converged, it = False, 0
    pi_old = g / g.sum(1, keepdims=True)
    for it in range(1, n_iter + 1):
        Elogpi = digamma(g) - digamma(g.sum(1, keepdims=True))
        d = digamma(lam0 + lam1)
        Elb, El1b = digamma(lam1) - d, digamma(lam0) - d
        s = np.where(y[:, None], (Elb - log_eps)[None, :], (El1b - log_1meps)[None, :])
        Ea, Eb = Elogpi[a], Elogpi[b]
        for _ in range(inner):
            phi_ab = _softmax(Ea + phi_ba * s)
            phi_ba = _softmax(Eb + phi_ab * s)
        g = scatter(phi_ab, phi_ba)
        pp = (phi_ab * phi_ba) * w[:, None]
        lam1 = eta1 + pp[y].sum(0)
        lam0 = eta0 + pp[~y].sum(0)
        pi = g / g.sum(1, keepdims=True)
        delta = float(np.abs(pi - pi_old).mean())
        pi_old = pi
        if delta < tol:
            converged = True
            break
    bk = lam1 / (lam0 + lam1)
    same = pi_old[a] * pi_old[b]
    p1 = np.clip((same * bk[None, :]).sum(1) + (1.0 - same.sum(1)) * eps, EPS, 1 - EPS)
    loglik = float((w * np.where(y, np.log(p1), np.log1p(-p1))).sum())
    return {"pi": pi_old, "g": g, "b": bk, "lam": np.stack([lam1, lam0], 1), "eps": eps, "loglik": loglik,
            "n_iter": it, "converged": converged, "K": K, "seed": seed, "n_pairs": P, "init": init}


def heldout_loglik(fit: dict, m: int, edges, pairs) -> float:
    """Mean log-likelihood per held-out pair (pairs = two index arrays), with E[pi] and E[b] plugged in."""
    elin = _norm_edges(edges, m)
    hl = _norm_edges(pairs, m)
    a, b = hl // m, hl % m
    y = np.isin(hl, elin)
    pi, bk, eps = fit["pi"], fit["b"], fit["eps"]
    same = pi[a] * pi[b]                                  # P(z_ab = z_ba = k)
    p1 = (same * bk[None, :]).sum(1) + (1.0 - same.sum(1)) * eps
    p1 = np.clip(p1, EPS, 1 - EPS)
    return float(np.where(y, np.log(p1), np.log1p(-p1)).mean())
