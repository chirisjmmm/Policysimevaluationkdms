"""D1 Response Variance (variance decomposition), D2 Semantic Dispersion
(sec 5.4). D3 Pareto is cross-cutting (D1/D2 vs X1/A1) and is assembled at
report-build time from already-computed metrics, not here.

D1 note: role_slot ("AIResearcher#1") is NOT a fixed set across runs -- each
run's persona composition varies (e.g. one run has 3 AIResearchers + 1
AIPractitioner, another has 2+2). So the role factor used for the
between-role / within-role decomposition is `stakeholder_type`, which *is*
a consistent category across runs; role_slot's #N suffix still disambiguates
individuals within a (run, stakeholder_type) group.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.io.ir import PHASE_ORDER


def _predicted_vars_in_phase(phase: str, var_types: dict) -> list:
    return [v for (p, v), r in var_types.items() if p == phase and r.var_type == "predicted"]


def run_d1(runs: list, var_types: dict) -> dict:
    """Nested variance decomposition: Total = between_run + between_role_within_run
    + within_role_across_runs (residual), pooled over all (phase, predicted var)."""
    by_phase = {}
    grand_micro, grand_role, grand_run, grand_total = [], [], [], []

    for phase in PHASE_ORDER:
        pred_vars = _predicted_vars_in_phase(phase, var_types)
        if not pred_vars:
            continue
        ss_run = ss_role = ss_resid = ss_total = 0.0
        n_total = 0
        for var in pred_vars:
            all_vals = []
            run_records = []
            for run in runs:
                try:
                    ph = run.phase(phase)
                except KeyError:
                    continue
                by_role: dict = defaultdict(list)
                for post in ph.posts:
                    if var in post.prediction_values:
                        by_role[post.stakeholder_type].append(post.prediction_values[var])
                if not by_role:
                    continue
                run_vals = [v for vs in by_role.values() for v in vs]
                all_vals.extend(run_vals)
                run_records.append((run_vals, by_role))
            if len(all_vals) < 2:
                continue
            grand_mean = np.mean(all_vals)
            for run_vals, by_role in run_records:
                run_mean = np.mean(run_vals)
                ss_run += len(run_vals) * (run_mean - grand_mean) ** 2
                for role_vals in by_role.values():
                    role_mean = np.mean(role_vals)
                    ss_role += len(role_vals) * (role_mean - run_mean) ** 2
                    ss_resid += sum((v - role_mean) ** 2 for v in role_vals)
            ss_total += sum((v - grand_mean) ** 2 for v in all_vals)
            n_total += len(all_vals)

        if n_total < 2 or ss_total == 0:
            continue
        var_run = ss_run / n_total
        var_role = ss_role / n_total
        var_resid = ss_resid / n_total
        var_total = ss_total / n_total
        by_phase[phase] = {
            "var_within_role_across_runs": var_resid, "var_between_role_within_run": var_role,
            "var_between_run": var_run, "var_total": var_total,
            "ICC_role": var_role / var_total if var_total else None,
            "ICC_run": var_run / var_total if var_total else None,
            "n_obs": n_total,
        }
        grand_micro.append(var_resid)
        grand_role.append(var_role)
        grand_run.append(var_run)
        grand_total.append(var_total)

    d1 = float(np.mean(grand_micro)) if grand_micro else None
    icc_role = float(np.mean([r["ICC_role"] for r in by_phase.values() if r["ICC_role"] is not None])) if by_phase else None
    icc_run = float(np.mean([r["ICC_run"] for r in by_phase.values() if r["ICC_run"] is not None])) if by_phase else None
    return {"D1": d1, "ICC_role": icc_role, "ICC_run": icc_run, "by_phase": by_phase}


# --------------------------------------------------------------------------
# D2: Semantic Dispersion
# --------------------------------------------------------------------------

def _ngrams(text: str, n: int) -> list:
    tokens = text.lower().split()
    return [tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def _dispersion(narratives: list) -> dict | None:
    narratives = [n for n in narratives if n and n.strip()]
    if len(narratives) < 2:
        return None
    vec = TfidfVectorizer()
    m = vec.fit_transform(narratives)
    sims = cosine_similarity(m)
    n = len(narratives)
    pairwise = [1 - sims[i, j] for i in range(n) for j in range(i + 1, n)]
    bigrams = [g for t in narratives for g in _ngrams(t, 2)]
    trigrams = [g for t in narratives for g in _ngrams(t, 3)]
    return {
        "mean_pairwise_distance": float(np.mean(pairwise)),
        "distinct_2": (len(set(bigrams)) / len(bigrams)) if bigrams else None,
        "distinct_3": (len(set(trigrams)) / len(trigrams)) if trigrams else None,
    }


def run_d2(runs: list, backend: str = "tfidf") -> dict:
    by_phase = {p: [] for p in PHASE_ORDER}
    for run in runs:
        for ph in run.phases:
            r = _dispersion([p.narrative for p in ph.posts])
            if r is not None:
                by_phase[ph.phase].append(r)

    by_phase_agg = {}
    all_dist, all_d2, all_d3 = [], [], []
    for phase, records in by_phase.items():
        if not records:
            continue
        dists = [r["mean_pairwise_distance"] for r in records]
        d2s = [r["distinct_2"] for r in records if r["distinct_2"] is not None]
        d3s = [r["distinct_3"] for r in records if r["distinct_3"] is not None]
        by_phase_agg[phase] = {
            "mean_pairwise_distance": float(np.mean(dists)),
            "distinct_2": float(np.mean(d2s)) if d2s else None,
            "distinct_3": float(np.mean(d3s)) if d3s else None,
            "n_run_phase_obs": len(records),
        }
        all_dist.extend(dists)
        all_d2.extend(d2s)
        all_d3.extend(d3s)

    return {
        "D2": float(np.mean(all_dist)) if all_dist else None,
        "distinct_2": float(np.mean(all_d2)) if all_d2 else None,
        "distinct_3": float(np.mean(all_d3)) if all_d3 else None,
        "backend": backend,
        "by_phase": by_phase_agg,
    }
