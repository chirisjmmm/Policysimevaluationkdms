"""B1 -- Branch Label Fidelity (v4 guideline sec 8.5, branch-pathway only).

Checks whether the enabling/baseline/constraining assumption labels agree in
sign with their simulated outcomes (enabling >= baseline >= constraining,
direction-adjusted) across the 40 sibling triples (= completed_parent_ids),
with a Friedman test since the triples are a paired/blocked design (same
policy, same variable, three assumption conditions per parent node).

The guideline explicitly warns this can come out sign-reversed (NSA: median
emergency-OON-reduction was enabling 8.0 / baseline 8.5 / constraining 11.5
-- "constraining" showed the BIGGEST improvement). This module never
silently assumes higher_is_better is the correct direction: it always
computes violations under BOTH directions and flags when the opposite
direction fits better, so a direction mix-up surfaces as data rather than
getting baked into a single directional score.
"""
from __future__ import annotations

import statistics

from scipy.stats import friedmanchisquare

TRANSITIONS = ("enabling", "baseline", "constraining")


def _triple_values(tree, target_var: str) -> list:
    triples = []
    for parent in tree.meta.get("completed_parent_ids", []):
        vals = {}
        for t in TRANSITIONS:
            key = f"{parent}/{t}" if parent != "root" else f"root/{t}"
            node = tree.nodes.get(key)
            if node is None:
                continue
            posts = node["phase_result"]["initial_posts"]
            values = [p["prediction_values"][target_var] for p in posts if target_var in p["prediction_values"]]
            if values:
                vals[t] = statistics.mean(values)
        if len(vals) == 3:
            triples.append(vals)
    return triples


def _monotone(v: dict, direction: str) -> bool:
    e, b, c = v["enabling"], v["baseline"], v["constraining"]
    return (e >= b >= c) if direction == "higher_is_better" else (e <= b <= c)


def label_fidelity(tree, target_var: str, direction: str = "higher_is_better") -> dict:
    triples = _triple_values(tree, target_var)
    if not triples:
        return {"status": "NO_DATA", "target_var": target_var}

    violations_configured = sum(1 for v in triples if not _monotone(v, direction))
    opposite = "lower_is_better" if direction == "higher_is_better" else "higher_is_better"
    violations_opposite = sum(1 for v in triples if not _monotone(v, opposite))

    stat = p = None
    if len(triples) >= 2:
        stat, p = friedmanchisquare(
            [t["enabling"] for t in triples], [t["baseline"] for t in triples],
            [t["constraining"] for t in triples],
        )
        stat, p = float(stat), float(p)

    medians = {t: statistics.median(tri[t] for tri in triples) for t in TRANSITIONS}

    direction_warning = None
    if violations_opposite < violations_configured:
        direction_warning = (
            f"'{opposite}' 방향이 위반 건수가 더 적음 (configured={violations_configured}/{len(triples)} "
            f"vs opposite={violations_opposite}/{len(triples)}). direction 설정이 실제 라벨 의미와 "
            f"반대일 가능성 -- medians 확인 후 direction을 재확정할 것."
        )

    return {
        "status": "OK", "target_var": target_var, "direction": direction,
        "B1": 1 - violations_configured / len(triples),
        "n_violations": violations_configured, "n_triples": len(triples),
        "friedman_chi2": stat, "p": p, "medians": medians,
        "direction_warning": direction_warning,
        "n_violations_opposite_direction": violations_opposite,
    }


def label_fidelity_all_outcomes_vars(tree, outcomes_vars: list, direction: str = "higher_is_better") -> dict:
    return {var: label_fidelity(tree, var, direction) for var in outcomes_vars}
