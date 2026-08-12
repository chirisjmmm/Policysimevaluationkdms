"""Cell-vs-cell contrasts with bootstrap CIs (sec 6.1-6.3).

Design is an incomplete 2x2 (see configs/cells.yaml header for the
guideline-vs-reality correction: bk21 data is DeepSeek, not Gemini). Never
pool raw data across cells (sec 6.2 rule 1) -- contrasts always compare two
whole cells' worth of per-run statistics, computed independently.

Bootstrap CIs are computed by resampling *precomputed per-run scalar values*
(the standard efficient nonparametric bootstrap for a cell-level statistic),
not by re-running the full NLP pipeline per resample. This is provided only
for metrics with a natural per-run decomposition (X1 APE, per-run A1 score,
per-run-phase M1 convergence, per-run-phase D2 dispersion) -- other metrics
(A2, DM1, DM2, X2-X5) are reported as single-point cell comparisons without
a bootstrap CI, which is a documented scope limitation, not an oversight.
"""
from __future__ import annotations

import numpy as np


def bootstrap_delta(values_a, values_b, n_resamples: int = 500, seed: int = 20260810) -> dict:
    a = np.asarray([v for v in values_a if v is not None], dtype=float)
    b = np.asarray([v for v in values_b if v is not None], dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"a_mean": None, "b_mean": None, "delta": None, "delta_CI95": None, "n_a": len(a), "n_b": len(b)}

    rng = np.random.default_rng(seed)
    deltas = np.empty(n_resamples)
    for i in range(n_resamples):
        sa = a[rng.integers(0, len(a), len(a))]
        sb = b[rng.integers(0, len(b), len(b))]
        deltas[i] = sa.mean() - sb.mean()
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "a_mean": float(a.mean()), "b_mean": float(b.mean()), "delta": float(a.mean() - b.mean()),
        "delta_CI95": [float(lo), float(hi)], "n_a": len(a), "n_b": len(b),
    }


def stability_curve(values, n_points: list, n_resamples: int = 500, seed: int = 20260810) -> dict:
    """Learning-curve check (sec 6.3): does the cell-level mean stabilize by n=50?"""
    arr = np.asarray([v for v in values if v is not None], dtype=float)
    rng = np.random.default_rng(seed)
    out = {}
    for n in n_points:
        if n > len(arr):
            continue
        means = np.empty(n_resamples)
        for i in range(n_resamples):
            sample = arr[rng.choice(len(arr), size=n, replace=False if n <= len(arr) else True)]
            means[i] = sample.mean()
        out[str(n)] = {"mean": float(means.mean()), "std": float(means.std())}
    return out
