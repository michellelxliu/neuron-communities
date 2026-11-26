"""Method 1: mixed membership on the prompt x neuron matrix X (n prompts x m neurons of one parent community).

Model: x_pn ~ Poisson(mu_pn), mu_pn = theta_p . beta_n, with theta (n x K) and beta (m x K) nonnegative. theta_p is how
much each of the K sub-communities is in use on prompt p; beta_n is how much neuron n takes part in each. Maximum
likelihood by the multiplicative (EM) updates of KL-NMF (Lee & Seung 2001), which never decrease the likelihood.

link="bernoulli" is for 0/1 data: x_pn = 1[c_pn > 0] with a latent count c_pn ~ Poisson(mu_pn), so
P(x = 1) = 1 - exp(-mu) (the Bernoulli-Poisson link of BigCLAM and of HPF on binary data). EM: the E-step replaces each
observed 1 by E[c | x = 1] = mu / (1 - exp(-mu)), then the Poisson M-step runs on those expected counts.

Continuous nonnegative entries (attribution shares) are fit by the same updates; the Poisson log-likelihood is then a
quasi-likelihood. The fit is scale-equivariant (scaling X scales theta), but log-likelihood values depend on the scale,
so compare K only on the same X.

Held-out cells (``heldout``: row and column index arrays, zeros included) are left out of the fit: they enter neither
the ratio term nor the normalising sums. Score them with ``heldout_loglik``.
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.special import gammaln

EPS = 1e-12
CHUNK = 1 << 20          # nonzeros per chunk when computing mu at the observed cells


def _csr(X) -> sparse.csr_matrix:
    X = sparse.csr_matrix(X, dtype=np.float64)
    X.sum_duplicates()
    X.eliminate_zeros()
    if X.nnz and X.data.min() < 0:
        raise ValueError("X must be nonnegative")
    return X


def _mu_at(theta, beta, r, c):
    out = np.empty(len(r))
    for s in range(0, len(r), CHUNK):
        e = s + CHUNK
        out[s:e] = np.einsum("ik,ik->i", theta[r[s:e]], beta[c[s:e]])
    return out


def _cells_matrix(cells, shape) -> sparse.csr_matrix | None:
    if cells is None:
        return None
    hr, hc = (np.asarray(a, dtype=np.int64) for a in cells)
    M = sparse.csr_matrix((np.ones(len(hr)), (hr, hc)), shape=shape)
    M.sum_duplicates()
    M.data[:] = 1.0
    return M


def _loglik_obs(x, mu_nz, total_mu, link):
    """Log-likelihood over the observed cells: x, mu_nz at the observed nonzeros; total_mu = sum of mu over all
    observed cells (nonzero or not)."""
    if link == "poisson":
        return float(np.sum(x * np.log(mu_nz + EPS)) - total_mu - np.sum(gammaln(x + 1.0)))
    # bernoulli: log(1 - exp(-mu)) at the ones, -mu at the zeros
    return float(np.sum(np.log(-np.expm1(-mu_nz) + EPS)) - (total_mu - mu_nz.sum()))


def fit_pf(X, K: int, *, link: str = "poisson", heldout=None, n_iter: int = 2000, tol: float = 1e-6,
           seed: int = 0, check_every: int = 10) -> dict:
    """Fit theta (n x K) and beta (m x K) to X. Returns theta, beta (columns of beta sum to 1, scale moved into theta),
    loglik (observed cells, final), trace (loglik every ``check_every`` iterations), n_iter, converged."""
    if link not in ("poisson", "bernoulli"):
        raise ValueError(link)
    X = _csr(X)
    if link == "bernoulli" and X.nnz and not np.all(X.data == 1.0):
        raise ValueError("link='bernoulli' needs 0/1 data")
    n, m = X.shape
    M = _cells_matrix(heldout, X.shape)
    obs = X if M is None else (X - X.multiply(M)).tocsr()
    obs.eliminate_zeros()
    coo = obs.tocoo()
    r, c, x = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data

    rng = np.random.default_rng(seed)
    mean_x = max(x.sum() / max(n * m - (0 if M is None else M.nnz), 1), EPS)
    s0 = np.sqrt(mean_x / K)
    theta = rng.gamma(1.0, s0, size=(n, K)) + EPS
    beta = rng.gamma(1.0, s0, size=(m, K)) + EPS

    def masked_sums(theta, beta):
        """sum of mu over all observed cells."""
        tot = float(theta.sum(0) @ beta.sum(0))
        if M is not None:
            mc = M.tocoo()
            tot -= float(_mu_at(theta, beta, mc.row.astype(np.int64), mc.col.astype(np.int64)).sum())
        return tot

    def ratio_matrix(mu_nz):
        xe = x if link == "poisson" else mu_nz / np.maximum(-np.expm1(-mu_nz), EPS)   # E[c | x = 1]
        return sparse.csr_matrix((xe / np.maximum(mu_nz, EPS), (r, c)), shape=(n, m))

    trace, prev, converged, it = [], None, False, 0
    for it in range(1, n_iter + 1):
        R = ratio_matrix(_mu_at(theta, beta, r, c))
        den = beta.sum(0)[None, :] - (M @ beta if M is not None else 0.0)
        theta *= (R @ beta) / np.maximum(den, EPS)
        R = ratio_matrix(_mu_at(theta, beta, r, c))
        den = theta.sum(0)[None, :] - (M.T @ theta if M is not None else 0.0)
        beta *= (R.T @ theta) / np.maximum(den, EPS)
        if it % check_every == 0 or it == n_iter:
            ll = _loglik_obs(x, _mu_at(theta, beta, r, c), masked_sums(theta, beta), link)
            trace.append(ll)
            if prev is not None and abs(ll - prev) <= tol * max(abs(prev), 1.0):
                converged = True
                break
            prev = ll
    s = beta.sum(0)
    s[s <= 0] = 1.0
    beta = beta / s
    theta = theta * s
    return {"theta": theta, "beta": beta, "loglik": trace[-1], "trace": np.array(trace), "n_iter": it,
            "converged": converged, "link": link, "K": K, "seed": seed}


def heldout_loglik(fit: dict, X, cells) -> float:
    """Mean log-likelihood per held-out cell (cells = row and column index arrays, as passed to ``fit_pf``)."""
    X = _csr(X)
    hr, hc = (np.asarray(a, dtype=np.int64) for a in cells)
    lin = np.unique(hr * X.shape[1] + hc)
    hr, hc = lin // X.shape[1], lin % X.shape[1]
    x = np.asarray(X[hr, hc]).ravel()
    mu = _mu_at(fit["theta"], fit["beta"], hr, hc)
    if fit["link"] == "poisson":
        ll = x * np.log(mu + EPS) - mu - gammaln(x + 1.0)
    else:
        ll = np.where(x > 0, np.log(-np.expm1(-mu) + EPS), -mu)
    return float(ll.mean())


def transform(X, beta: np.ndarray, *, link: str = "poisson", n_iter: int = 500, tol: float = 1e-6,
              seed: int = 0) -> np.ndarray:
    """theta for new prompts (e.g. held-out ones) with beta fixed: the theta half of the updates only."""
    X = _csr(X)
    n, m = X.shape
    coo = X.tocoo()
    r, c, x = coo.row.astype(np.int64), coo.col.astype(np.int64), coo.data
    K = beta.shape[1]
    rng = np.random.default_rng(seed)
    theta = rng.gamma(1.0, max(x.sum() / max(n * K, 1), EPS), size=(n, K)) + EPS
    den = np.maximum(beta.sum(0)[None, :], EPS)
    for _ in range(n_iter):
        mu = _mu_at(theta, beta, r, c)
        xe = x if link == "poisson" else mu / np.maximum(-np.expm1(-mu), EPS)
        R = sparse.csr_matrix((xe / np.maximum(mu, EPS), (r, c)), shape=(n, m))
        new = theta * (R @ beta) / den
        done = np.max(np.abs(new - theta)) <= tol * max(np.max(np.abs(theta)), 1.0)
        theta = new
        if done:
            break
    return theta
