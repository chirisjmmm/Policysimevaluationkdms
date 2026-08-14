"""Does predicted-value direction split by stakeholder role in
nsa_single_deepseek? Ad-hoc analysis, not part of the pipeline.

Real stakeholder_type categories in this dataset (checked directly against
the raw scenario JSON, not assumed): only two exist --
  PolicyRole        (occupation: registered_nurse / physician /
                      licensed_practical_or_licensed_vocational_nurse)
                     -- clinical/provider-side workers
  RegulatoryAgency   (occupation: management_analyst / compliance_officer /
                      lawyer) -- regulatory/administrative/legal workers
There is no payer, patient, or mediator stakeholder_type in this data (that
taxonomy belongs to a different policy/dataset). D1's ICC_role (v4
guideline) also uses `stakeholder_type` as its role factor (see
src/metrics/div_micro.py run_d1 docstring), so this is the same "role" axis
already summarized in the 9-metric scorecard as D1=0.192.
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import yaml
from scipy.stats import mannwhitneyu, ttest_ind

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cli import build_policy_context
from src.io.loaders import load_cell
from src.prep.var_typing import classify_all_variables

eval_cfg = yaml.safe_load(open(ROOT / "configs/eval_default.yaml", encoding="utf-8"))
config, doc_index, kg_index = build_policy_context("nsa_single", eval_cfg)
runs, _ = load_cell(str(ROOT / "data/policies/nsa/runs/deepseek_v4flash"), "nsa_single_deepseek")
var_types = classify_all_variables(runs, config, doc_index, kg_index)

PREDICTED_VARS = sorted({(p, v) for (p, v), r in var_types.items() if r.var_type == "predicted"})

# ---------------------------------------------------------------------
# 1. Per stakeholder_type mean/std for every predicted (phase, var)
# ---------------------------------------------------------------------
by_role_var = defaultdict(list)   # (phase, var, stakeholder_type) -> [values]
narratives_by_role_var = defaultdict(list)  # (phase, var, stakeholder_type) -> [(agent_key, narrative)]

for run in runs:
    for ph in run.phases:
        for post in ph.posts:
            for (p, v) in PREDICTED_VARS:
                if p != ph.phase:
                    continue
                if v in post.prediction_values:
                    by_role_var[(p, v, post.stakeholder_type)].append(post.prediction_values[v])
                    narratives_by_role_var[(p, v, post.stakeholder_type)].append((post.agent_key, post.narrative))

roles = sorted({k[2] for k in by_role_var})
print("stakeholder_type categories actually present:", roles)
print()

summary_rows = []
for (p, v) in PREDICTED_VARS:
    row = {"phase": p, "var": v}
    for role in roles:
        vals = by_role_var.get((p, v, role), [])
        row[f"{role}_mean"] = statistics.mean(vals) if vals else None
        row[f"{role}_std"] = statistics.pstdev(vals) if len(vals) > 1 else None
        row[f"{role}_n"] = len(vals)
    summary_rows.append(row)

print(f"{'phase':10s} {'var':55s} " + " | ".join(f"{r:>28s}" for r in roles))
for row in summary_rows:
    cells = []
    for role in roles:
        m, s, n = row[f"{role}_mean"], row[f"{role}_std"], row[f"{role}_n"]
        cells.append(f"mean={m:8.2f} sd={s:8.2f} n={n:3d}" if m is not None else "n/a")
    print(f"{row['phase']:10s} {row['var'][:55]:55s} " + " | ".join(cells))

# ---------------------------------------------------------------------
# 2+3. Focus on the two X1 targets: bar-chart data + t-test / rank-sum
# ---------------------------------------------------------------------
TARGETS = ["reduction_emergency_oon_bills_pct", "reduction_nonemergency_oon_bills_pct"]
GT = {"reduction_emergency_oon_bills_pct": 15, "reduction_nonemergency_oon_bills_pct": 11}

print("\n=== X1 target variables: PolicyRole(provider) vs RegulatoryAgency(regulator) ===")
target_results = {}
for var in TARGETS:
    a = by_role_var.get(("Outcomes", var, "PolicyRole"), [])
    b = by_role_var.get(("Outcomes", var, "RegulatoryAgency"), [])
    t_stat, t_p = ttest_ind(a, b, equal_var=False)
    u_stat, u_p = mannwhitneyu(a, b, alternative="two-sided")
    direction = "PolicyRole(provider) > RegulatoryAgency(regulator)" if statistics.mean(a) > statistics.mean(b) \
        else "RegulatoryAgency(regulator) > PolicyRole(provider)"
    target_results[var] = {
        "GT": GT[var],
        "PolicyRole_mean": statistics.mean(a), "PolicyRole_std": statistics.pstdev(a), "PolicyRole_n": len(a),
        "RegulatoryAgency_mean": statistics.mean(b), "RegulatoryAgency_std": statistics.pstdev(b), "RegulatoryAgency_n": len(b),
        "welch_t": t_stat, "welch_p": t_p,
        "mannwhitney_u": u_stat, "mannwhitney_p": u_p,
        "direction": direction,
        "diff": statistics.mean(a) - statistics.mean(b),
    }
    print(f"\n{var} (GT={GT[var]})")
    print(f"  PolicyRole(provider):      n={len(a):3d} mean={statistics.mean(a):.2f} sd={statistics.pstdev(a):.2f}")
    print(f"  RegulatoryAgency(regulator): n={len(b):3d} mean={statistics.mean(b):.2f} sd={statistics.pstdev(b):.2f}")
    print(f"  Welch t={t_stat:.3f}, p={t_p:.4f}  |  Mann-Whitney U={u_stat:.1f}, p={u_p:.4f}")
    print(f"  direction: {direction}  (diff={statistics.mean(a)-statistics.mean(b):+.2f}pp)")

# ---------------------------------------------------------------------
# 4. Role-explained variance for just these two variables (ICC_role
# analog restricted to Outcomes phase, these 2 vars) + narrative quotes
# mentioning budget/administrative/patient-burden language, split by role
# and by whether the post over- or under-shoots GT.
# ---------------------------------------------------------------------
def icc_role_for_var(phase, var):
    by_role = defaultdict(list)
    all_vals = []
    for run in runs:
        try:
            ph = run.phase(phase)
        except KeyError:
            continue
        for post in ph.posts:
            if var in post.prediction_values:
                by_role[post.stakeholder_type].append(post.prediction_values[var])
                all_vals.append(post.prediction_values[var])
    grand_mean = np.mean(all_vals)
    ss_role = sum(len(vals) * (np.mean(vals) - grand_mean) ** 2 for vals in by_role.values())
    ss_total = sum((v - grand_mean) ** 2 for v in all_vals)
    return ss_role / ss_total if ss_total else None

print("\n=== role-explained variance (ICC_role analog) for the two X1 target vars ===")
for var in TARGETS:
    icc = icc_role_for_var("Outcomes", var)
    print(f"  {var}: ICC_role={icc:.3f}")

KEYWORDS = ["budget", "cost", "administrative", "burden", "expense", "resource", "funding",
            "compliance cost", "patient", "out-of-pocket", "premium", "enforcement"]

print("\n=== sample narrative excerpts mentioning burden/cost language, by role ===")
for var in TARGETS:
    print(f"\n--- {var} ---")
    for role in ["PolicyRole", "RegulatoryAgency"]:
        quotes = narratives_by_role_var.get(("Outcomes", var, role), [])
        shown = 0
        for agent, narrative in quotes:
            low = (narrative or "").lower()
            if any(k in low for k in KEYWORDS) and shown < 2:
                # print a short window around the first keyword hit
                idx = min((low.find(k) for k in KEYWORDS if k in low), default=0)
                snippet = narrative[max(0, idx - 80):idx + 220].replace("\n", " ")
                print(f"  [{role} / {agent}] ...{snippet}...")
                shown += 1

out = {
    "roles_present": roles,
    "summary_rows": summary_rows,
    "target_results": target_results,
    "icc_role_targets": {v: icc_role_for_var("Outcomes", v) for v in TARGETS},
}
out_path = ROOT / "outputs" / "nsa_single_role_direction.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False, default=str)
print(f"\nwrote {out_path}")
