"""Planted overlapping structure for tests and checks of both methods.

groups: list of neuron index arrays over 0..m-1; a neuron in two groups is a planted shared neuron.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse


def bipartite(n_prompts: int, m: int, groups, *, p_second: float = 0.3, rate: float = 3.0,
              background: float = 0.002, binary: bool = False, seed: int = 0):
    """Prompt x neuron counts from the Method 1 model. Each prompt uses one group, and with probability p_second a
    second one; theta ~ Gamma(5, rate / 5) on the groups it uses. beta_k is spread over group k's neurons with
    Gamma(5, 1/5) heterogeneity and sums to 1 over the group. x ~ Poisson(theta beta^T + background); binary=True
    returns 1[x > 0]. Returns (X csr, theta, beta)."""
    rng = np.random.default_rng(seed)
    K = len(groups)
    beta = np.zeros((m, K))
    for k, g in enumerate(groups):
        v = rng.gamma(5.0, 0.2, size=len(g))
        beta[np.asarray(g), k] = v / v.sum()
    theta = np.zeros((n_prompts, K))
    first = rng.integers(0, K, size=n_prompts)
    theta[np.arange(n_prompts), first] = rng.gamma(5.0, rate / 5.0, size=n_prompts) * len(groups[0])
    two = rng.random(n_prompts) < p_second
    second = (first + rng.integers(1, K, size=n_prompts)) % K
    theta[np.flatnonzero(two), second[two]] = rng.gamma(5.0, rate / 5.0, size=int(two.sum())) * len(groups[0])
    X = rng.poisson(theta @ beta.T + background).astype(float)
    if binary:
        X = (X > 0).astype(float)
    return sparse.csr_matrix(X), theta, beta


def graph(m: int, groups, *, p_in: float = 0.5, eps: float = 0.01, seed: int = 0):
    """Edges from the a-MMSB generative model: a neuron in one group has pi = that group; a neuron in several groups
    splits pi evenly; neurons in no group are spread evenly over all groups. Returns (edges as (a, b) arrays, pi)."""
    rng = np.random.default_rng(seed)
    K = len(groups)
    pi = np.zeros((m, K))
    for k, g in enumerate(groups):
        pi[np.asarray(g), k] = 1.0
    pi[pi.sum(1) == 0] = 1.0
    pi /= pi.sum(1, keepdims=True)
    a, b = np.triu_indices(m, 1)
    za = np.array([rng.choice(K, p=pi[i]) for i in a])
    zb = np.array([rng.choice(K, p=pi[j]) for j in b])
    p = np.where(za == zb, p_in, eps)
    keep = rng.random(len(a)) < p
    return (a[keep], b[keep]), pi
