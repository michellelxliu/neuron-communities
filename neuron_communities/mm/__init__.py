"""Mixed-membership community detection inside a Leiden community (plan: stat236 docs/plans/
2026-11-26-within-community-mixed-membership.md).

  pf       Method 1: Poisson factorization of the prompt x neuron matrix (link "poisson"), or the Bernoulli-Poisson link
           for 0/1 data (link "bernoulli", P(x = 1) = 1 - exp(-mu)), fit by EM / multiplicative updates.
  mmsb     Method 2: assortative mixed-membership stochastic blockmodel on the community's co-occurrence graph,
           fit by variational inference.
  select   held-out cells and pairs, choice of K, stability across seeds, membership sets and shared neurons.
  planted  planted overlapping data for tests and checks.
"""
