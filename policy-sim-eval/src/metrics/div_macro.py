"""DX1 Outcome Support Width, DX2 Bin Distribution Entropy, DX3 Trajectory
Clustering (sec 5.6). DX4 (branch contribution decomposition) is branch-only
and not applicable -- no branch/assumption-conditioned data in this dataset.
"""
from __future__ import annotations

import statistics

import numpy as np
from scipy.stats import entropy
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.io.ir import PHASE_ORDER


def _phase_var_mean(run, phase: str, var: str):
    try:
        ph = run.phase(phase)
    except KeyError:
        return None
    values = [p.prediction_values[var] for p in ph.posts if var in p.prediction_values]
    return statistics.mean(values) if values else None


# --------------------------------------------------------------------------
# DX1: Outcome Support Width (Diversity -- macro; v4 primary score = CV)
#
# CV = sigma / |mean| is the score reported alongside X4 (calibration):
# small DX1 + coverage=False + PIT in {0,1} tells the causal story "the run
# ensemble explored too narrow a range of outcomes to ever have bracketed
# the true value" -- diversity failure as the cause of an accuracy failure.
# range/IQR/std are still reported as descriptive detail underneath it.
# --------------------------------------------------------------------------

def run_dx1(runs: list, targets: list) -> dict:
    out = {}
    for t in targets:
        var = t["name"]
        vals = np.array([r.target_values[var] for r in runs if var in r.target_values], dtype=float)
        if len(vals) < 2:
            out[var] = None
            continue
        q1, q3 = np.percentile(vals, [25, 75])
        mean = float(vals.mean())
        out[var] = {
            "CV": float(vals.std() / max(abs(mean), 1e-9)),
            "range": float(vals.max() - vals.min()), "iqr": float(q3 - q1),
            "std": float(vals.std()), "mean": mean, "min": float(vals.min()), "max": float(vals.max()),
            "n": len(vals),
        }
    return out


# --------------------------------------------------------------------------
# DX2: Bin Distribution Entropy (relative to GT, when available)
# --------------------------------------------------------------------------

_BIN_EDGES = [-np.inf, -50, -20, 20, 50, np.inf]
_BIN_LABELS = ["far_under", "under", "near", "over", "far_over"]


def run_dx2(runs: list, targets: list) -> dict:
    out = {}
    for t in targets:
        var, true_value = t["name"], t.get("true_value")
        if true_value is None:
            out[var] = {"status": "GT_UNAVAILABLE"}
            continue
        vals = np.array([r.target_values[var] for r in runs if var in r.target_values], dtype=float)
        if len(vals) < 2:
            out[var] = {"status": "NO_PREDICTIONS"}
            continue
        rel_err = (vals - true_value) / max(abs(true_value), 1e-9) * 100
        bin_idx = np.digitize(rel_err, _BIN_EDGES[1:-1])
        counts = np.bincount(bin_idx, minlength=len(_BIN_LABELS))
        probs = counts[counts > 0] / counts.sum()
        h = entropy(probs, base=np.e)
        out[var] = {
            "status": "OK", "entropy": float(h), "entropy_norm": float(h / np.log(len(_BIN_LABELS))),
            "bin_counts": dict(zip(_BIN_LABELS, counts.tolist())), "n": len(vals),
        }
    return out


# --------------------------------------------------------------------------
# DX3: Trajectory Clustering
# --------------------------------------------------------------------------

def run_dx3(runs: list, var_types: dict, k_range: tuple = (2, 8), seed: int = 20260810) -> dict:
    keys = sorted({(p, v) for (p, v), r in var_types.items() if r.var_type == "predicted"})
    if not keys:
        return {"status": "NO_PREDICTED_VARS"}

    features, run_ids = [], []
    for run in runs:
        row = []
        ok = True
        for phase, var in keys:
            v = _phase_var_mean(run, phase, var)
            if v is None:
                ok = False
                break
            row.append(v)
        if ok:
            features.append(row)
            run_ids.append(run.run_id)

    if len(features) < max(k_range[0] + 1, 6):
        return {"status": "INSUFFICIENT_RUNS", "n_runs": len(features)}

    X = np.array(features)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    Xz = (X - X.mean(axis=0)) / std

    n_components = min(len(keys), len(features) - 1, 10)
    pca = PCA(n_components=n_components, random_state=seed)
    Xp = pca.fit_transform(Xz)

    best_k, best_score, best_labels = None, -2.0, None
    max_k = min(k_range[1], len(features) - 1)
    for k in range(k_range[0], max_k + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=10).fit_predict(Xp)
        if len(set(labels)) < 2:
            continue
        score = silhouette_score(Xp, labels)
        if score > best_score:
            best_k, best_score, best_labels = k, score, labels

    if best_labels is None:
        return {"status": "NO_STABLE_CLUSTERING", "n_runs": len(features)}

    cluster_sizes = {int(c): int((best_labels == c).sum()) for c in set(best_labels)}
    return {
        "status": "OK", "k": int(best_k), "silhouette": float(best_score),
        "cluster_sizes": cluster_sizes,
        "explained_variance_ratio": pca.explained_variance_ratio_[:3].tolist(),
        "n_runs": len(features), "n_features": len(keys),
    }
