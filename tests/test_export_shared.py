"""Pruned export: per layer max(top 50, |s| >= 0.1 x max), a prefix of the stored entries, with n_kept."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts" / "export"))
from export_shared import prune_table  # noqa: E402


def test_prune_table():
    rng = np.random.default_rng(0)
    n, L, K = 7, 3, 200
    s = rng.standard_normal((n, L, K)).astype(np.float32) * np.exp(-np.arange(K) / 20)[None, None, :]
    order = np.argsort(-np.abs(s), axis=2)
    s = np.take_along_axis(s, order, 2).astype(np.float32)                    # descending |s| as stored
    idx = rng.integers(0, 14336, (n, L, K)).astype(np.uint16)
    h = rng.standard_normal((n, L, K)).astype(np.float16)
    s[0, 1, :120] = np.sign(s[0, 1, :120]) * np.float32(1.0)                             # many neurons at the max: keep 120 > 50
    lst = lambda arr, typ: pa.ListArray.from_arrays(pa.array(np.arange(n + 1, dtype=np.int32) * L * K), pa.array(arr.ravel(), typ))
    t = pa.table({"manifest_index": pa.array(np.arange(n)), "idx": lst(idx, pa.uint16()), "s": lst(s, pa.float32()),
                  "h": lst(h, pa.float16())})
    out = prune_table(t, L, K)
    nk = np.stack(out.column("n_kept").to_numpy(zero_copy_only=False))
    assert nk.shape == (n, L) and nk.min() >= 50 and nk[0, 1] == 120
    for i in range(n):
        si = out.column("s")[i].values.to_numpy(); ii = out.column("idx")[i].values.to_numpy()
        assert len(si) == nk[i].sum()
        off = 0
        for l in range(L):
            k = nk[i, l]
            assert np.array_equal(si[off:off + k], s[i, l, :k]) and np.array_equal(ii[off:off + k], idx[i, l, :k])
            a = np.abs(s[i, l])
            assert k == max(50, int((a >= 0.1 * a[0]).sum()))
            off += k
