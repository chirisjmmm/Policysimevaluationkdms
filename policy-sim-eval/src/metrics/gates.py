"""G1/G2/G3 data-validity gates (v4 guideline sec 5, "게이트 3종").

Pass/fail lines, not scores -- v3 scored chain consistency (X2), concept-
duplicate internal agreement (X3_internal), and value-range plausibility
(X5) as if they were accuracy metrics. They aren't: none of them compare
against an external ground truth, they only check whether the simulation's
own numbers are internally coherent. v4 demotes all three to gates reported
as a single pass/fail line each, freeing "X1/X4" to mean exactly one thing
(agreement with real-world ground truth).
"""
from __future__ import annotations

import ast
import operator
from collections import defaultdict

from src.metrics.acc_macro import _phase_var_mean

_CMP_OPS = {ast.LtE: operator.le, ast.GtE: operator.ge, ast.Lt: operator.lt,
            ast.Gt: operator.gt, ast.Eq: operator.eq, ast.NotEq: operator.ne}


def _resolve_operand(node, values: dict):
    if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
        return values.get(f"{node.value.id}.{node.attr}")
    if isinstance(node, ast.Constant):
        return node.value
    raise ValueError(f"unsupported constraint operand: {ast.dump(node)}")


def _eval_constraint(expr: str, values: dict):
    tree = ast.parse(expr, mode="eval").body
    if not isinstance(tree, ast.Compare) or len(tree.ops) != 1:
        raise ValueError(f"unsupported constraint expression (must be a single comparison): {expr}")
    left = _resolve_operand(tree.left, values)
    right = _resolve_operand(tree.comparators[0], values)
    if left is None or right is None:
        return None
    return _CMP_OPS[type(tree.ops[0])](left, right)


# --------------------------------------------------------------------------
# G1: Chain consistency (arithmetic chain_constraints + concept_duplicate
# value agreement, excluding headcount_mislabeled members)
# --------------------------------------------------------------------------

def run_g1(runs: list, chain_constraints: list, concept_duplicates: list | None = None,
           var_types: dict | None = None, agreement_rel_tol: float = 0.10) -> dict:
    violations = evaluated = 0
    by_constraint = {c: {"violations": 0, "evaluated": 0} for c in chain_constraints}
    for run in runs:
        tokens = set()
        for c in chain_constraints:
            for node in ast.walk(ast.parse(c, mode="eval")):
                if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
                    tokens.add((node.value.id, node.attr))
        values = {f"{ph}.{var}": _phase_var_mean(run, ph, var) for ph, var in tokens}
        for c in chain_constraints:
            result = _eval_constraint(c, values)
            if result is None:
                continue
            evaluated += 1
            by_constraint[c]["evaluated"] += 1
            if not result:
                violations += 1
                by_constraint[c]["violations"] += 1

    agreement_violations = agreement_evaluated = 0
    by_group = {}
    if concept_duplicates:
        for group in concept_duplicates:
            members = [m for m in group["members"]]
            if var_types is not None:
                kept = []
                for m in members:
                    phase, var = m.split(".")
                    vt = var_types.get((phase, var))
                    if vt is not None and vt.var_type == "headcount_mislabeled":
                        continue
                    kept.append(m)
                members = kept
            if len(members) < 2:
                by_group[group["group"]] = {"n_evaluated": 0, "n_violations": 0,
                                              "note": "fewer than 2 comparable (non-headcount) members"}
                continue
            g_eval = g_viol = 0
            for run in runs:
                vals = []
                for m in members:
                    phase, var = m.split(".")
                    v = _phase_var_mean(run, phase, var)
                    if v is not None:
                        vals.append(v)
                if len(vals) < 2:
                    continue
                g_eval += 1
                rel_diff = abs(vals[0] - vals[1]) / max(abs(vals[0]), abs(vals[1]), 1e-9)
                if rel_diff > agreement_rel_tol:
                    g_viol += 1
            by_group[group["group"]] = {"n_evaluated": g_eval, "n_violations": g_viol}
            agreement_evaluated += g_eval
            agreement_violations += g_viol

    total_evaluated = evaluated + agreement_evaluated
    total_violations = violations + agreement_violations
    return {
        "pass": (total_violations == 0) if total_evaluated else None,
        "n_violations": total_violations, "n_evaluated": total_evaluated,
        "chain": {"n_violations": violations, "n_evaluated": evaluated, "by_constraint": by_constraint},
        "concept_duplicate_agreement": {
            "n_violations": agreement_violations, "n_evaluated": agreement_evaluated,
            "rel_tol": agreement_rel_tol, "by_group": by_group,
        },
    }


# --------------------------------------------------------------------------
# G2: Value range plausibility (excludes headcount_mislabeled variables --
# there is no reliable range to check them against once we know they hold
# counts, not the percentages their names imply)
# --------------------------------------------------------------------------

def run_g2(runs: list, value_ranges: dict, var_types: dict | None = None) -> dict:
    violations = total = 0
    by_var = defaultdict(lambda: {"violations": 0, "total": 0})
    for phase_var, rng in value_ranges.items():
        phase, var = phase_var.split(".")
        if var_types is not None:
            vt = var_types.get((phase, var))
            if vt is not None and vt.var_type == "headcount_mislabeled":
                continue
        for run in runs:
            try:
                ph = run.phase(phase)
            except KeyError:
                continue
            for post in ph.posts:
                if var not in post.prediction_values:
                    continue
                v = post.prediction_values[var]
                total += 1
                by_var[phase_var]["total"] += 1
                lo = rng.get("min", float("-inf"))
                hi = rng.get("max", float("inf"))
                is_violation = not (lo <= v <= hi)
                if rng.get("integer") and float(v) != int(v):
                    is_violation = True
                if is_violation:
                    violations += 1
                    by_var[phase_var]["violations"] += 1

    violating_vars = [v for v, d in by_var.items() if d["violations"] > 0]
    return {
        "pass": (violations == 0) if total else None,
        "n_violations": violations, "n_evaluated": total,
        "violating_vars": violating_vars, "by_var": dict(by_var),
    }


# --------------------------------------------------------------------------
# G3: Output completion rate (parsing failures / nulls)
# --------------------------------------------------------------------------

def run_g3(runs: list, var_types: dict) -> dict:
    expected = defaultdict(int)
    actual = defaultdict(int)
    for run in runs:
        for ph in run.phases:
            phase_vars = {v for (p, v) in var_types if p == ph.phase}
            for post in ph.posts:
                for var in phase_vars:
                    expected[(ph.phase, var)] += 1
                    if var in post.prediction_values and post.prediction_values[var] is not None:
                        actual[(ph.phase, var)] += 1

    total_expected = sum(expected.values())
    total_actual = sum(actual.values())
    by_var = {
        f"{phase}.{var}": {"expected": n, "actual": actual.get((phase, var), 0),
                            "completion_rate": actual.get((phase, var), 0) / n}
        for (phase, var), n in expected.items() if actual.get((phase, var), 0) < n
    }
    completion_rate = (total_actual / total_expected) if total_expected else None
    return {
        "pass": (completion_rate == 1.0) if completion_rate is not None else None,
        "completion_rate": completion_rate, "n_expected": total_expected, "n_actual": total_actual,
        "incomplete_vars": by_var,
    }
