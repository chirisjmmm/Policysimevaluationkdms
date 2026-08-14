"""Diagnostic: is nsa_single_deepseek's low M1 (0.152) real disagreement or
parsing/missingness noise? Ad-hoc analysis script, not part of the pipeline.

1. Per-run completeness flag (canonical-schema field presence, not the
   noise-polluted var_types pool used by G3) + M1 recomputed on the
   complete-only subset vs the full 50.
2. Missingness broken down by phase and by canonical variable, including
   whether a "missing" field is actually present under a known typo alias
   (compliant_notice_provision_rate <-> complaint_notice_provision_rate).
3. Variable-name character length vs missing rate, for the scatter plot.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.cli import build_policy_context
from src.io.loaders import load_cell
from src.metrics.acc_meso import run_m1
from src.prep.var_typing import classify_all_variables

RUNS_DIR = "data/policies/nsa/runs/deepseek_v4flash"
CELL_ID = "nsa_single_deepseek"
POLICY_ID = "nsa_single"

eval_cfg = yaml.safe_load(open(ROOT / "configs/eval_default.yaml", encoding="utf-8"))
config, doc_index, kg_index = build_policy_context(POLICY_ID, eval_cfg)
runs, load_report = load_cell(str(ROOT / RUNS_DIR), CELL_ID)
var_types = classify_all_variables(runs, config, doc_index, kg_index)

# ---------------------------------------------------------------------
# Canonical schema (from the source data's own inner_schema.json, not my
# variable_anchors declaration -- this is the "ground truth" of what each
# post in a given phase was actually asked to produce).
# ---------------------------------------------------------------------
schema_path = (ROOT / ".." / "HY-UP_Simulation_Results" /
               "20260712131323_deepseek-v4-flash_result_NSA_fwd_fwdbwd_sc1-50" /
               "20260712131323_deepseek-v4-flash_result_NSA_fwd_fwdbwd_sc1-50" / "inner_schema.json")
schema = json.load(open(schema_path, encoding="utf-8"))
canonical_by_phase = {p["phase"]: [f["name"] for f in p["prediction_fields"]] for p in schema["phases"]}

KNOWN_ALIASES = {
    "compliant_notice_provision_rate": ["complaint_notice_provision_rate"],
}

# ---------------------------------------------------------------------
# Full M1 (sanity check against evaluation.json's 0.152)
# ---------------------------------------------------------------------
m1_full = run_m1(runs, var_types, eval_cfg["diversity"]["cv_ref"])

# ---------------------------------------------------------------------
# Per-run completeness scan
# ---------------------------------------------------------------------
run_missing = {}          # run_id -> {"missing": n, "expected": n, "detail": [...]}
missing_by_phase_var = {} # (phase, var) -> {"missing": n, "expected": n, "recovered_via_alias": n}

for run in runs:
    total_missing = 0
    total_expected = 0
    detail = []
    for ph in run.phases:
        canon_vars = canonical_by_phase.get(ph.phase, [])
        for post in ph.posts:
            for var in canon_vars:
                key = (ph.phase, var)
                d = missing_by_phase_var.setdefault(key, {"missing": 0, "expected": 0, "recovered_via_alias": 0})
                total_expected += 1
                d["expected"] += 1
                if var in post.prediction_values and post.prediction_values[var] is not None:
                    continue
                # missing under canonical name -- check known alias
                alias_hit = any(a in post.prediction_values and post.prediction_values[a] is not None
                                 for a in KNOWN_ALIASES.get(var, []))
                total_missing += 1
                d["missing"] += 1
                if alias_hit:
                    d["recovered_via_alias"] += 1
                detail.append((ph.phase, var, post.agent_key, "alias_present" if alias_hit else "truly_missing"))
    run_missing[run.run_id] = {"missing": total_missing, "expected": total_expected, "detail": detail}

n_complete = sum(1 for r in run_missing.values() if r["missing"] == 0)
complete_run_ids = {rid for rid, r in run_missing.items() if r["missing"] == 0}
complete_runs = [r for r in runs if r.run_id in complete_run_ids]
incomplete_runs = [r for r in runs if r.run_id not in complete_run_ids]

m1_complete = run_m1(complete_runs, var_types, eval_cfg["diversity"]["cv_ref"]) if complete_runs else None
m1_incomplete = run_m1(incomplete_runs, var_types, eval_cfg["diversity"]["cv_ref"]) if incomplete_runs else None

# ---------------------------------------------------------------------
# Phase-level missingness
# ---------------------------------------------------------------------
phase_totals = {}
for (phase, var), d in missing_by_phase_var.items():
    pt = phase_totals.setdefault(phase, {"missing": 0, "expected": 0})
    pt["missing"] += d["missing"]
    pt["expected"] += d["expected"]

# ---------------------------------------------------------------------
# Variable-level missingness (for scatter: name length vs missing rate)
# ---------------------------------------------------------------------
var_level = []
for (phase, var), d in missing_by_phase_var.items():
    rate = d["missing"] / d["expected"] if d["expected"] else None
    truly_missing = d["missing"] - d["recovered_via_alias"]
    truly_missing_rate = truly_missing / d["expected"] if d["expected"] else None
    var_level.append({
        "phase": phase, "var": var, "name_len": len(var), "expected": d["expected"],
        "missing": d["missing"], "missing_rate": rate,
        "recovered_via_alias": d["recovered_via_alias"],
        "truly_missing": truly_missing, "truly_missing_rate": truly_missing_rate,
    })
var_level.sort(key=lambda x: -x["missing_rate"])

# Pearson/Spearman correlation: name_len vs missing_rate (both raw and
# alias-corrected "truly missing" rate)
import numpy as np
from scipy.stats import pearsonr, spearmanr

lens = np.array([v["name_len"] for v in var_level], dtype=float)
rates = np.array([v["missing_rate"] for v in var_level], dtype=float)
true_rates = np.array([v["truly_missing_rate"] for v in var_level], dtype=float)

pear_r, pear_p = pearsonr(lens, rates)
spear_r, spear_p = spearmanr(lens, rates)
pear_r_true, pear_p_true = pearsonr(lens, true_rates)
spear_r_true, spear_p_true = spearmanr(lens, true_rates)

# ---------------------------------------------------------------------
# Per-run M1 distribution: complete vs incomplete, plus raw CV inspection
# to see if disagreement (not missingness) is the driver even among
# complete runs.
# ---------------------------------------------------------------------
from src.metrics.acc_meso import _run_level_m1

m1_per_complete_run = []
for r in complete_runs:
    m1_run, trend_run, by_phase = _run_level_m1(r, var_types, eval_cfg["diversity"]["cv_ref"])
    if m1_run is not None:
        m1_per_complete_run.append(m1_run)

m1_per_incomplete_run = []
for r in incomplete_runs:
    m1_run, trend_run, by_phase = _run_level_m1(r, var_types, eval_cfg["diversity"]["cv_ref"])
    if m1_run is not None:
        m1_per_incomplete_run.append(m1_run)

out = {
    "n_runs": len(runs),
    "m1_full": m1_full["M1"],
    "n_complete_runs": n_complete,
    "n_incomplete_runs": len(runs) - n_complete,
    "m1_complete_subset": m1_complete["M1"] if m1_complete else None,
    "m1_incomplete_subset": m1_incomplete["M1"] if m1_incomplete else None,
    "m1_per_complete_run_stats": {
        "mean": statistics.mean(m1_per_complete_run) if m1_per_complete_run else None,
        "median": statistics.median(m1_per_complete_run) if m1_per_complete_run else None,
        "min": min(m1_per_complete_run) if m1_per_complete_run else None,
        "max": max(m1_per_complete_run) if m1_per_complete_run else None,
        "n": len(m1_per_complete_run),
    },
    "m1_per_incomplete_run_stats": {
        "mean": statistics.mean(m1_per_incomplete_run) if m1_per_incomplete_run else None,
        "median": statistics.median(m1_per_incomplete_run) if m1_per_incomplete_run else None,
        "n": len(m1_per_incomplete_run),
    },
    "phase_missingness": {
        p: {"missing": d["missing"], "expected": d["expected"], "missing_rate": d["missing"] / d["expected"]}
        for p, d in sorted(phase_totals.items())
    },
    "var_level": var_level,
    "correlation_namelen_vs_missingrate": {
        "pearson_r": pear_r, "pearson_p": pear_p,
        "spearman_r": spear_r, "spearman_p": spear_p,
    },
    "correlation_namelen_vs_truly_missing_rate": {
        "pearson_r": pear_r_true, "pearson_p": pear_p_true,
        "spearman_r": spear_r_true, "spearman_p": spear_p_true,
    },
    "run_missing_counts": {rid: {"missing": r["missing"], "expected": r["expected"]} for rid, r in run_missing.items()},
}

out_path = ROOT / "outputs" / "nsa_single_m1_diagnostic.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)

print(json.dumps({k: v for k, v in out.items() if k not in ("run_missing_counts", "var_level")}, indent=2, ensure_ascii=False))
print("\n--- var_level (sorted by missing_rate desc) ---")
for v in var_level:
    print(f"{v['phase']:10s} {v['var']:70s} len={v['name_len']:3d} "
          f"missing={v['missing']:3d}/{v['expected']:3d} ({v['missing_rate']:.1%}) "
          f"truly_missing_rate={v['truly_missing_rate']:.1%} alias_recovered={v['recovered_via_alias']}")
print(f"\nwrote {out_path}")
