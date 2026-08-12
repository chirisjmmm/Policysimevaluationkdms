"""X1 Target Error, X4 Calibration (Accuracy -- Macro, v4 guideline sec 5).

GT exists only at Outcomes (eval terminal); Impact is post-target
plausibility only, never accuracy. X2 (chain), X3_internal (concept-dup
agreement), X5 (value range) moved to gates.py as pass/fail gates in v4 --
they measure internal consistency and plausibility, not accuracy against a
ground truth, so scoring them as if they were accuracy metrics overstated
what they show. X3_upstream is absorbed into this file's run_x1 instead of
being a separate table, per v4 sec "X3_upstream → X1에 흡수".
"""
from __future__ import annotations

import statistics

import numpy as np

_PHASE_VAR_KEY_MISSING = object()


def _phase_var_mean(run, phase: str, var: str):
    try:
        ph = run.phase(phase)
    except KeyError:
        return None
    values = [p.prediction_values[var] for p in ph.posts if var in p.prediction_values]
    return statistics.mean(values) if values else None


def _bootstrap_ape(ape: np.ndarray, n_resamples: int, rng) -> tuple:
    boot_means = np.empty(n_resamples)
    for i in range(n_resamples):
        idx = rng.integers(0, len(ape), len(ape))
        boot_means[i] = ape[idx].mean()
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return float(lo), float(hi)


# --------------------------------------------------------------------------
# X1: Target Error (+ concept-duplicate upstream rows absorbed in, per v4)
# --------------------------------------------------------------------------

def run_x1(runs: list, targets: list, concept_duplicates: list | None = None,
           var_types: dict | None = None, n_resamples: int = 500, seed: int = 20260810) -> dict:
    rng = np.random.default_rng(seed)
    out = {}

    def _score(phase: str, var: str, true_value: float, source: str, source_meta: dict | None = None) -> dict:
        # phase-aware lookup, not run.target_values -- concept-duplicate upstream
        # members are frequently in a DIFFERENT phase than Outcomes (e.g.
        # Outputs.approved_100e_projects), and target_values only ever holds
        # Outcomes-phase aggregates. Using target_values here silently produced
        # NO_PREDICTIONS for every non-Outcomes upstream member (found via
        # direct inspection of the v4 report before this fix).
        yhat = np.array([v for r in runs if (v := _phase_var_mean(r, phase, var)) is not None], dtype=float)
        if len(yhat) == 0:
            return {"status": "NO_PREDICTIONS", "source": source, **(source_meta or {})}
        ape = np.abs(yhat - true_value) / max(abs(true_value), 1e-9) * 100
        lo, hi = _bootstrap_ape(ape, n_resamples, rng)
        return {
            "status": "OK", "true_value": true_value, "source": source,
            "APE_mean": float(ape.mean()), "APE_median": float(np.median(ape)),
            "CI95": [lo, hi], "n": len(yhat), **(source_meta or {}),
        }

    target_lookup = {}
    for t in targets:
        var, true_value, phase = t["name"], t.get("true_value"), t.get("phase", "Outcomes")
        if true_value is None:
            out[var] = {"status": "GT_UNAVAILABLE", "source": "direct", "gt_source_note": t.get("source")}
            continue
        target_lookup[var] = true_value
        out[var] = _score(phase, var, true_value, "direct", {"gt_source_note": t.get("source")})

    if concept_duplicates:
        for group in concept_duplicates:
            gt_key = group["ground_truth_from"]
            gt_var = gt_key.split(".")[-1]
            true_value = target_lookup.get(gt_var)
            if true_value is None:
                continue
            for member in group["members"]:
                phase, var = member.split(".")
                if var == gt_var or var in out:
                    continue
                if var_types is not None:
                    vt = var_types.get((phase, var))
                    if vt is not None and vt.var_type == "headcount_mislabeled":
                        continue  # v4 sec 3.4: excluded from GT comparison entirely
                out[var] = _score(phase, var, true_value, "concept_duplicate", {"group": group["group"]})

    return out


# --------------------------------------------------------------------------
# X4: Calibration (Outcomes only)
# --------------------------------------------------------------------------

def run_x4(runs: list, targets: list) -> dict:
    out = {}
    for t in targets:
        var, true_value = t["name"], t.get("true_value")
        yhat = np.array([r.target_values[var] for r in runs if var in r.target_values], dtype=float)
        if true_value is None or len(yhat) == 0:
            out[var] = {"status": "GT_UNAVAILABLE" if true_value is None else "NO_PREDICTIONS"}
            continue
        lo50, hi50 = np.percentile(yhat, [25, 75])
        lo90, hi90 = np.percentile(yhat, [5, 95])
        pit = float(np.mean(yhat <= true_value))
        term1 = float(np.mean(np.abs(yhat - true_value)))
        term2 = 0.5 * float(np.mean(np.abs(yhat[:, None] - yhat[None, :])))
        out[var] = {
            "status": "OK",
            "coverage_50": bool(lo50 <= true_value <= hi50),
            "coverage_90": bool(lo90 <= true_value <= hi90),
            "PIT": pit, "CRPS": term1 - term2, "n": len(yhat),
        }
    return out
