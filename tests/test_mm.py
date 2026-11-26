"""Mixed-membership methods on planted overlapping data (CPU, small): Method 1 (Poisson factorization, both links) and
Method 2 (a-MMSB) must recover planted groups and their shared neurons; the selection helpers must behave."""
from __future__ import annotations

import numpy as np
import pytest

from neuron_communities.mm import mmsb, pf, planted, select

GROUPS = [np.arange(0, 12), np.arange(8, 20), np.arange(20, 30)]          # shared: 8..11
SHARED = set(range(8, 12))


def _best_jaccards(sets, groups):
    return [max(select.jaccard(s, g) for s in sets) for g in groups]


def _pf_stable(X, K, link, seeds=range(4)):
    return select.stable_fit(lambda s: (lambda f: (f["beta"], f["loglik"], f))(pf.fit_pf(X, K, link=link, seed=s)),
                             seeds)


def test_pf_loglik_never_decreases():
    X, _, _ = planted.bipartite(500, 30, GROUPS, seed=0)
    f = pf.fit_pf(X, 3, n_iter=200, tol=0, check_every=1)
    assert np.all(np.diff(f["trace"]) >= -1e-6 * np.abs(f["trace"][:-1]))
    assert np.allclose(f["beta"].sum(0), 1.0)


@pytest.mark.parametrize("link,binary", [("poisson", False), ("bernoulli", True)])
def test_pf_recovers_planted_overlap(link, binary):
    X, _, _ = planted.bipartite(1500, 30, GROUPS, binary=binary, seed=1)
    res = _pf_stable(X, 3, link)
    assert res["stable"].all()
    sets = select.membership_sets(res["loadings"])
    assert min(_best_jaccards(sets, GROUPS)) >= 0.75
    shared = set(select.summarize(sets, 30, res["stable"])["shared"])
    assert shared <= SHARED and len(shared) >= 3


def test_pf_choose_k():
    X, _, _ = planted.bipartite(1500, 30, GROUPS, seed=2)
    cells = select.sample_cells(X.shape, 0.1, seed=7)
    scores = {K: [pf.heldout_loglik(pf.fit_pf(X, K, heldout=cells, seed=s), X, cells) for s in range(2)]
              for K in (2, 3, 4)}
    assert select.choose_k(scores)[0] == 3


def test_pf_heldout_cells_are_left_out():
    """Changing X only at held-out cells must not change the fit."""
    X, _, _ = planted.bipartite(400, 30, GROUPS, seed=3)
    cells = select.sample_cells(X.shape, 0.1, seed=1)
    X2 = X.tolil()
    X2[cells[0], cells[1]] = 7.0
    f1 = pf.fit_pf(X, 3, heldout=cells, seed=0, n_iter=50, tol=0)
    f2 = pf.fit_pf(X2.tocsr(), 3, heldout=cells, seed=0, n_iter=50, tol=0)
    assert np.allclose(f1["beta"], f2["beta"]) and np.allclose(f1["theta"], f2["theta"])


def test_pf_transform_matches_fit():
    X, _, _ = planted.bipartite(800, 30, GROUPS, seed=4)
    f = pf.fit_pf(X, 3, seed=0)
    th = pf.transform(X, f["beta"])
    # same beta -> theta of the fit's prompts is recovered (up to EM tolerance)
    assert np.corrcoef(th.ravel(), f["theta"].ravel())[0, 1] > 0.99


GROUPS_G = [np.arange(0, 30), np.arange(22, 52), np.arange(52, 80)]      # shared: 22..29
SHARED_G = set(range(22, 30))


def test_mmsb_recovers_planted_overlap():
    E, _ = planted.graph(80, GROUPS_G, p_in=0.6, eps=0.01, seed=2)
    res = select.stable_fit(lambda s: (lambda f: (f["pi"], f["loglik"], f))(mmsb.fit_mmsb(80, E, 3, seed=s)),
                            range(4))
    assert res["stable"].all()
    sets = select.membership_sets(res["loadings"])
    assert min(_best_jaccards(sets, GROUPS_G)) >= 0.75
    shared = set(select.summarize(sets, 80, res["stable"])["shared"])
    assert len(shared & SHARED_G) >= 5 and len(shared - SHARED_G) <= 4


def test_mmsb_choose_k():
    E, _ = planted.graph(80, GROUPS_G, p_in=0.6, eps=0.01, seed=3)
    hp = select.sample_pairs(80, 0.1, seed=3)
    scores = {K: [mmsb.heldout_loglik(mmsb.fit_mmsb(80, E, K, heldout=hp, seed=s), 80, E, hp) for s in range(2)]
              for K in (2, 3, 4)}
    assert select.choose_k(scores)[0] == 3


def test_mmsb_subsampled_nonedges_weighting():
    """With non-edges subsampled, the weights restore the non-edge count."""
    E, _ = planted.graph(80, GROUPS_G, seed=4)
    a, b, y, w = mmsb.build_pairs(80, E, max_pairs=1500, seed=0)
    assert np.all(a < b) and y.sum() == len(E[0])
    assert np.isclose(w[~y].sum(), 80 * 79 // 2 - len(E[0]))
    assert len(np.unique(a * 80 + b)) == len(a)


def test_sample_pairs_and_cells():
    a, b = select.sample_pairs(50, 0.2, seed=0)
    assert np.all(a < b) and np.all(b < 50) and len(np.unique(a * 50 + b)) == len(a) == 245
    r, c = select.sample_cells((40, 30), 0.1, seed=0)
    assert len(np.unique(r * 30 + c)) == len(r) == 120


def test_membership_and_summary():
    L = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0], [0.001, 0.0]])
    sets = select.membership_sets(L)
    assert [s.tolist() for s in sets] == [[0, 1], [1, 2]]
    s = select.summarize(sets, 4)
    assert s["shared"] == [1] and s["unassigned"] == [3] and s["pairs"] == [{"a": 0, "b": 1, "shared": [1]}]
