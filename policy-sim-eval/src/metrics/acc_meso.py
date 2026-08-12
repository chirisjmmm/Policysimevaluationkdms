"""M1 Convergence, M2 Value Uptake (v4 guideline sec 5).

Both operate on `predicted` variables only -- anchored_constant variance is
measurement error, not signal; headcount_mislabeled variables are excluded
too (a "rate" variable actually holding headcounts contributes no genuine
convergence/uptake signal, only noise from a units bug).

M3 (anchoring coefficient) was deleted in v4: forcing a per-variable
z-score with only ~5 observations mechanically induces negative correlation
between an agent's value and the mean of its predecessors' z-scores (an
artifact of the standardization, not a measured social-anchoring effect),
and the single-shot narration structure gives too little data to interpret
it as social anchoring even if the artifact weren't there.
"""
from __future__ import annotations

import statistics

import numpy as np
from scipy.stats import spearmanr

from src.io.ir import PHASE_ORDER
from src.io.normalize import extract_quantities


def _predicted_vars_in_phase(phase: str, var_types: dict) -> list:
    return [v for (p, v), r in var_types.items() if p == phase and r.var_type == "predicted"]


def _cv(values: list) -> float:
    if len(values) < 2:
        return 0.0
    mean = statistics.mean(values)
    std = statistics.pstdev(values)
    return std / max(abs(mean), 1e-9)


# --------------------------------------------------------------------------
# M1: Convergence
#
# v4 fix: aggregation must be identical whichever way run_m1 is called --
# once over a whole cell (50 runs) for the cell-summary number, and once
# per-run (via a 1-run call) for contrasts/stability curves. v3 broke this:
# the cell-level path pooled every run's CVs together per phase before
# taking a median ("median of the pooled CV distribution"), while the
# contrast path averaged one-M1-per-run ("mean of per-run medians") -- two
# different statistics that happened to share a name, producing 0.978 in
# one report table and 0.844 in another for the same cell (bk21_deepseek).
# Fixed by always computing a single per-run M1 first, then macro-averaging
# across runs -- both call sites now literally reduce to the same formula.
# --------------------------------------------------------------------------

def _run_level_m1(run, var_types: dict, cv_ref: float) -> tuple:
    convergence_by_phase = {}
    for ph in run.phases:
        pred_vars = _predicted_vars_in_phase(ph.phase, var_types)
        cvs = []
        for var in pred_vars:
            values = [p.prediction_values[var] for p in ph.posts if var in p.prediction_values]
            if len(values) >= 2:
                cvs.append(_cv(values))
        if cvs:
            median_cv = statistics.median(cvs)
            convergence_by_phase[ph.phase] = 1 - min(max(median_cv / cv_ref, 0.0), 1.0)

    if not convergence_by_phase:
        return None, None, {}

    m1_run = statistics.mean(convergence_by_phase.values())
    trend_run = None
    ordered = [convergence_by_phase[p] for p in PHASE_ORDER if p in convergence_by_phase]
    if len(ordered) >= 3:
        rho, _ = spearmanr(range(len(ordered)), ordered)
        if not np.isnan(rho):
            trend_run = float(rho)
    return m1_run, trend_run, convergence_by_phase


def run_m1(runs: list, var_types: dict, cv_ref: float = 0.5) -> dict:
    m1_scores, trend_scores = [], []
    by_phase_acc = {p: [] for p in PHASE_ORDER}

    for run in runs:
        m1_run, trend_run, convergence_by_phase = _run_level_m1(run, var_types, cv_ref)
        if m1_run is not None:
            m1_scores.append(m1_run)
        if trend_run is not None:
            trend_scores.append(trend_run)
        for phase, conv in convergence_by_phase.items():
            by_phase_acc[phase].append(conv)

    by_phase = {
        phase: {"convergence_mean": float(np.mean(vs)), "n_runs": len(vs)}
        for phase, vs in by_phase_acc.items() if vs
    }
    m1 = float(np.mean(m1_scores)) if m1_scores else None
    m1_trend = float(np.mean(trend_scores)) if trend_scores else None
    return {"M1": m1, "M1_trend": m1_trend, "by_phase": by_phase, "n_runs_scored": len(m1_scores)}


# --------------------------------------------------------------------------
# M2: Value Uptake (numeric echo, not name citation)
# --------------------------------------------------------------------------

def _value_uptake_for_phase(phase) -> tuple:
    ordered_posts = sorted(phase.posts, key=lambda p: p.order_index)
    hits = trials = 0
    for k, post in enumerate(ordered_posts):
        if k == 0:
            continue
        prior_values = {
            v for p in ordered_posts[:k] for v in p.prediction_values.values()
            if isinstance(v, (int, float))
        }
        mine = {v for v in post.prediction_values.values() if isinstance(v, (int, float))}
        seen = {q.value for q in extract_quantities(post.narrative or "")}
        echoed = {v for v in seen & prior_values if v not in mine}
        hits += 1 if echoed else 0
        trials += 1
    return hits, trials


def run_m2(runs: list) -> dict:
    by_phase = {p: {"hits": 0, "trials": 0} for p in PHASE_ORDER}
    for run in runs:
        for ph in run.phases:
            hits, trials = _value_uptake_for_phase(ph)
            by_phase[ph.phase]["hits"] += hits
            by_phase[ph.phase]["trials"] += trials

    for phase, d in by_phase.items():
        d["value_uptake"] = (d["hits"] / d["trials"]) if d["trials"] else None

    total_hits = sum(d["hits"] for d in by_phase.values())
    total_trials = sum(d["trials"] for d in by_phase.values())
    m2 = (total_hits / total_trials) if total_trials else None
    return {"M2": m2, "by_phase": by_phase}
