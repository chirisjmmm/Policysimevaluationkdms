"""Main pipeline entrypoint. Produces outputs/evaluation.json (v4 guideline sec 9)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.branch.label_fidelity import label_fidelity
from src.branch.paths import compute_n_eff, extract_runs
from src.branch.tree_loader import load_branch_tree
from src.design.cells import load_cells_config
from src.design.contrasts import bootstrap_delta, stability_curve
from src.index.doc_index import DocIndex
from src.index.kg_index import build_kg_index
from src.index.retriever import DEFAULT_MULTILINGUAL_MODEL
from src.io.loaders import load_cell
from src.metrics.acc_macro import run_x1, run_x4
from src.metrics.acc_meso import run_m1, run_m2
from src.metrics.acc_micro import run_a1, run_a2
from src.metrics.div_macro import run_dx1, run_dx2, run_dx3
from src.metrics.div_meso import run_dm1, run_dm2
from src.metrics.div_micro import run_d1, run_d2
from src.metrics.gates import run_g1, run_g2, run_g3
from src.prep.doc_clean import normalize_with_map
from src.prep.var_typing import classify_all_variables

ROOT = Path(__file__).resolve().parent.parent


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _asdict_evidence(ev_list, limit=200):
    out = []
    for e in ev_list[:limit]:
        out.append({
            "fact_id": e.fact_id, "verdict": e.verdict, "cell_id": e.cell_id, "run_id": e.run_id,
            "phase": e.phase, "agent_key": e.agent_key, "role_slot": e.role_slot,
            "utterance_text": e.utterance_text, "context_text": e.context_text,
            "parsed": {"value": e.observed.value, "kind": e.observed.kind,
                       "currency": e.observed.currency, "bound": e.observed.bound} if e.observed else None,
            "reference": e.reference,
        })
    return out


def _per_run_a1_scores(runs, fact_table):
    return [run_a1([r], fact_table)["score"] for r in runs]


def _per_run_m1(runs, var_types):
    return [run_m1([r], var_types)["M1"] for r in runs]


def _per_run_d2(runs, backend):
    return [run_d2([r], backend)["D2"] for r in runs]


def build_policy_context(policy_id: str, eval_cfg: dict):
    config = yaml.safe_load(open(ROOT / f"configs/policy_{policy_id}.yaml", encoding="utf-8"))
    for k, v in eval_cfg.items():
        config.setdefault(k, v)

    raw = open(ROOT / config["doc_path"], encoding="utf-8").read()
    doc_index = DocIndex(normalize_with_map(raw), retriever_backend=eval_cfg["retriever"]["backend"],
                          embedding_model=eval_cfg["retriever"].get("embedding_model", DEFAULT_MULTILINGUAL_MODEL))
    raw_kg_path = ROOT / config.get("graph_path", "")
    if config.get("graph_path") and raw_kg_path.exists():
        raw_kg = json.load(open(raw_kg_path, encoding="utf-8"))
        kg_index = build_kg_index(raw_kg, doc_index,
                                   align_confidence_t2_min=eval_cfg["kg"].get("align_confidence_t2_min", 0.55))
    else:
        kg_index = None
    return config, doc_index, kg_index


def compute_cell_metrics(cid: str, runs: list, config: dict, var_types: dict,
                          doc_index, kg_index, eval_cfg: dict, load_report=None) -> dict:
    """Computes the v4 metric set for one cell's worth of Run objects. Shared
    between single-pathway cells and branch-pathway node-level extraction
    (leaf-81 for A1/A2/M1/M2/D1/D2/DM1) -- the metric functions themselves
    don't know or care whether a Run came from a scenario file or a branch
    tree path."""
    log(f"  A1")
    a1 = run_a1(runs, config["fact_table"])
    log(f"  A2 (slow: per-sentence retrieval)")
    a2 = run_a2(runs, doc_index, kg_index, eval_cfg)
    log(f"  M1/M2")
    m1 = run_m1(runs, var_types, eval_cfg["diversity"]["cv_ref"])
    m2 = run_m2(runs)
    log(f"  D1/D2")
    d1 = run_d1(runs, var_types)
    d2 = run_d2(runs, eval_cfg["retriever"]["backend"])
    log(f"  DM1/DM2 (influence graphs)")
    dm1 = run_dm1(runs, eval_cfg["meso"])
    dm2 = run_dm2(runs, var_types, eval_cfg["seed"])

    result = {
        "n_runs": len(runs),
        "A1": {"score": a1["score"], "consistent": a1["supported"], "contradiction": a1["contradicted"],
               "unscored": a1["unscored"], "by_fact": a1["by_fact"]},
        "A2": {"score": a2.grounded_total, "by_phase": a2.by_phase},
        "M1": {"score": m1["M1"], "trend": m1["M1_trend"], "by_phase": m1["by_phase"]},
        "M2": {"score": m2["M2"], "by_phase": m2["by_phase"]},
        "D1": {"ICC_role": d1["ICC_role"], "ICC_run": d1["ICC_run"], "by_phase": d1["by_phase"]},
        "D2": {"score": d2["D2"], "by_phase": d2["by_phase"]},
        "DM1": {"score": dm1["DM1"], "topology": dm1["topology_counts"],
                "anchor_role_dist": dm1["anchor_role_dist"]},
        "_appendix": {
            "distinct_2": d2["distinct_2"], "distinct_3": d2["distinct_3"],
            "DM2": {"score": dm2["DM2"], "by_phase": dm2["by_phase"]},
        },
        "_evidence_a1": a1["evidence"],
        "_influence_graphs": dm1.get("influence_graphs", {}),
    }
    if load_report is not None:
        result["load_report"] = {
            "revise_noop_verified": load_report.revise_noop_verified,
            "aggregated_value_matches_mean": load_report.aggregated_value_matches_mean,
            "n_revise_violations": len(load_report.revise_noop_violations),
            "n_agg_mismatches": len(load_report.aggregated_value_mismatches),
        }
    return result


def compute_cell_macro(cid: str, runs: list, config: dict, var_types: dict, eval_cfg: dict) -> dict:
    """X1/X4/DX1 (target-level metrics) + G1/G2/G3 (gates). For single-pathway
    cells `runs` is the same 50 runs as compute_cell_metrics; for branch
    cells use compute_target_metrics + compute_gates separately instead (see
    below) since the guideline specifies different sample-unit levels for
    each: X1/X4/DX1 at Outcomes-27, gates at the full leaf-81 node chains."""
    result = compute_target_metrics(runs, config, var_types, eval_cfg)
    result["gates"] = compute_gates(runs, config, var_types)
    return result


def compute_target_metrics(runs: list, config: dict, var_types: dict, eval_cfg: dict) -> dict:
    log(f"  X1/X4/DX1")
    x1 = run_x1(runs, config["targets"], config.get("concept_duplicates"), var_types,
                eval_cfg["bootstrap"]["n_resamples"], eval_cfg["seed"])
    x4 = run_x4(runs, config["targets"])
    dx1 = run_dx1(runs, config["targets"])

    log(f"  stability curves (X1 APE)")
    stability = {}
    for t in config["targets"]:
        if t.get("true_value") is None:
            continue
        var, true_value = t["name"], t["true_value"]
        yhat = [r.target_values[var] for r in runs if var in r.target_values]
        ape = [abs(y - true_value) / max(abs(true_value), 1e-9) * 100 for y in yhat]
        stability[var] = stability_curve(ape, eval_cfg["stability_curve"]["n_points"],
                                           eval_cfg["bootstrap"]["n_resamples"], eval_cfg["seed"])

    dx2 = run_dx2(runs, config["targets"])
    dx3 = run_dx3(runs, var_types, tuple(eval_cfg["diversity"]["kmeans_k_range"]), eval_cfg["seed"])

    return {"X1": x1, "X4": x4, "DX1": dx1, "stability_curve": stability, "_appendix": {"DX2": dx2, "DX3": dx3}}


def compute_gates(runs: list, config: dict, var_types: dict) -> dict:
    log(f"  gates G1/G2/G3")
    g1 = run_g1(runs, config["chain_constraints"], config.get("concept_duplicates"), var_types)
    g2 = run_g2(runs, config["value_ranges"], var_types)
    g3 = run_g3(runs, var_types)
    return {"G1": g1, "G2": g2, "G3": g3}


def run_pipeline():
    log("loading configs")
    eval_cfg = yaml.safe_load(open(ROOT / "configs/eval_default.yaml", encoding="utf-8"))
    cells_cfg = load_cells_config(str(ROOT / "configs/cells.yaml"))

    policy_ids = sorted({c["policy"] for c in cells_cfg["cells"]}
                         | {c["policy"] for c in cells_cfg.get("branch_cells", [])})
    policies = {}
    for policy_id in policy_ids:
        log(f"building policy context: {policy_id}")
        policies[policy_id] = build_policy_context(policy_id, eval_cfg)

    log("loading all cells")
    cell_runs = {}
    load_reports = {}
    for cell in cells_cfg["cells"]:
        runs, report = load_cell(str(ROOT / cell["runs_dir"]), cell["id"])
        cell_runs[cell["id"]] = runs
        load_reports[cell["id"]] = report
        log(f"  {cell['id']}: {len(runs)} runs loaded, revise_noop_verified={report.revise_noop_verified}, agg_matches_mean={report.aggregated_value_matches_mean}")

    log("classifying variables per policy (pooled across cells)")
    var_types_by_policy = {}
    for cell in cells_cfg["cells"]:
        pid = cell["policy"]
        if pid in var_types_by_policy:
            continue
        config, doc_index, kg_index = policies[pid]
        pooled_runs = [r for c in cells_cfg["cells"] if c["policy"] == pid for r in cell_runs[c["id"]]]
        var_types_by_policy[pid] = classify_all_variables(pooled_runs, config, doc_index, kg_index)

    by_cell = {}
    for cell in cells_cfg["cells"]:
        cid = cell["id"]
        pid = cell["policy"]
        config, doc_index, kg_index = policies[pid]
        var_types = var_types_by_policy[pid]
        runs = cell_runs[cid]
        log(f"=== computing metrics for cell {cid} (n={len(runs)}) ===")

        cell_result = compute_cell_metrics(cid, runs, config, var_types, doc_index, kg_index,
                                             eval_cfg, load_reports[cid])
        macro_result = compute_cell_macro(cid, runs, config, var_types, eval_cfg)
        cell_result.update(macro_result)
        by_cell[cid] = cell_result

    log("=== branch pathway cells ===")
    by_branch_cell = {}
    branch_leaf_runs = {}
    branch_var_types = {}
    for bcell in cells_cfg.get("branch_cells", []):
        cid = bcell["id"]
        pid = bcell["policy"]
        config, doc_index, kg_index = policies[pid]
        log(f"=== computing metrics for branch cell {cid} ===")

        tree = load_branch_tree(str(ROOT / bcell["tree_path"]), cid)
        leaf_runs = extract_runs(tree, "leaf")
        outcomes_runs = extract_runs(tree, "outcomes")
        log(f"  {cid}: {len(leaf_runs)} leaf runs, {len(outcomes_runs)} outcomes runs, "
            f"posting_order_fixed=True, extra_top_level_keys={tree.meta['extra_top_level_keys']}")

        var_types = classify_all_variables(leaf_runs, config, doc_index, kg_index)

        cell_result = compute_cell_metrics(cid, leaf_runs, config, var_types, doc_index, kg_index,
                                             eval_cfg, load_report=None)
        target_result = compute_target_metrics(outcomes_runs, config, var_types, eval_cfg)
        gates_result = compute_gates(leaf_runs, config, var_types)
        cell_result.update(target_result)
        cell_result["gates"] = gates_result

        log(f"  B1 label fidelity")
        outcomes_vars = sorted({v for run in outcomes_runs for v in run.target_values})
        # n_eff applies to the Outcomes-27 sample correlation itself, independent
        # of whether a target has GT (NSA's targets are both GT_UNAVAILABLE but
        # still deserve the correlation-correction caveat on their own X1/DX1 rows).
        n_eff = {var: compute_n_eff(outcomes_runs, var) for var in outcomes_vars}
        b1 = {var: label_fidelity(tree, var) for var in outcomes_vars}

        cell_result["n_outcomes_nodes"] = len(outcomes_runs)
        cell_result["n_eff"] = n_eff
        cell_result["n_leaves"] = len(leaf_runs)
        cell_result["n_triples"] = len(tree.meta.get("completed_parent_ids", []))
        cell_result["duplicate_narratives"] = tree.duplicate_narratives
        cell_result["posting_order_fixed"] = True
        cell_result["tree_meta"] = {
            "extra_top_level_keys": tree.meta["extra_top_level_keys"],
            "revise_noop_verified": tree.meta["revise_noop_verified"],
            "null_fields_verified": tree.meta["null_fields_verified"],
            "simulation_target": tree.meta.get("simulation_target"),
        }
        cell_result["B1"] = b1
        by_branch_cell[cid] = cell_result
        branch_leaf_runs[cid] = leaf_runs
        branch_var_types[cid] = var_types

    log("computing branch contrasts")
    branch_contrasts = {}
    for group_name, group in cells_cfg.get("branch_contrasts", {}).items():
        branch_contrasts[group_name] = {}
        for name, spec in group.items():
            cid_a, cid_b = spec["a"], spec["b"]
            runs_a, runs_b = branch_leaf_runs[cid_a], branch_leaf_runs[cid_b]
            m1_a = _per_run_m1(runs_a, branch_var_types[cid_a])
            m1_b = _per_run_m1(runs_b, branch_var_types[cid_b])
            branch_contrasts[group_name][name] = {
                "cell_a": cid_a, "cell_b": cid_b,
                "M1": bootstrap_delta(m1_a, m1_b, eval_cfg["bootstrap"]["n_resamples"], eval_cfg["seed"]),
                "D1_ICC_role": {"a": by_branch_cell[cid_a]["D1"]["ICC_role"],
                                 "b": by_branch_cell[cid_b]["D1"]["ICC_role"], "note": "point comparison, no bootstrap"},
                "DM1": {"a": by_branch_cell[cid_a]["DM1"]["score"],
                        "b": by_branch_cell[cid_b]["DM1"]["score"], "note": "point comparison, no bootstrap"},
                "note": spec.get("note"),
            }

    log("computing contrasts")
    contrasts = {}
    for name, spec in cells_cfg["contrasts"].items():
        if spec is None:
            contrasts[name] = None
            continue
        cid_a, cid_b = spec["a"], spec["b"]
        runs_a, runs_b = cell_runs[cid_a], cell_runs[cid_b]
        pid_a = next(c["policy"] for c in cells_cfg["cells"] if c["id"] == cid_a)
        pid_b = next(c["policy"] for c in cells_cfg["cells"] if c["id"] == cid_b)
        config_a = policies[pid_a][0]
        config_b = policies[pid_b][0]
        var_types_a = var_types_by_policy[pid_a]
        var_types_b = var_types_by_policy[pid_b]

        a1_a = _per_run_a1_scores(runs_a, config_a["fact_table"])
        a1_b = _per_run_a1_scores(runs_b, config_b["fact_table"])
        m1_a = _per_run_m1(runs_a, var_types_a)
        m1_b = _per_run_m1(runs_b, var_types_b)
        d2_a = _per_run_d2(runs_a, eval_cfg["retriever"]["backend"])
        d2_b = _per_run_d2(runs_b, eval_cfg["retriever"]["backend"])

        contrasts[name] = {
            "cell_a": cid_a, "cell_b": cid_b,
            "A1_score": bootstrap_delta(a1_a, a1_b, eval_cfg["bootstrap"]["n_resamples"], eval_cfg["seed"]),
            "M1": bootstrap_delta(m1_a, m1_b, eval_cfg["bootstrap"]["n_resamples"], eval_cfg["seed"]),
            "D2": bootstrap_delta(d2_a, d2_b, eval_cfg["bootstrap"]["n_resamples"], eval_cfg["seed"]),
        }
        if spec.get("confound_note"):
            contrasts[name]["confound_note"] = spec["confound_note"]
        # X1 contrast only when same target names exist in both (policy effect uses different targets -> skip)
        if pid_a == pid_b:
            for t in config_a["targets"]:
                if t.get("true_value") is None:
                    continue
                var, true_value = t["name"], t["true_value"]
                ape_a = [abs(r.target_values[var] - true_value) / max(abs(true_value), 1e-9) * 100
                         for r in runs_a if var in r.target_values]
                ape_b = [abs(r.target_values[var] - true_value) / max(abs(true_value), 1e-9) * 100
                         for r in runs_b if var in r.target_values]
                contrasts[name].setdefault("X1_APE", {})[var] = bootstrap_delta(
                    ape_a, ape_b, eval_cfg["bootstrap"]["n_resamples"], eval_cfg["seed"])

    evidence_all = []
    influence_graphs_all = {}
    for cid, data in list(by_cell.items()) + list(by_branch_cell.items()):
        evidence_all.extend(_asdict_evidence(data.pop("_evidence_a1")))
        influence_graphs_all.update(data.pop("_influence_graphs"))

    kg_meta = {pid: policies[pid][2].stats for pid in policies if policies[pid][2] is not None}

    all_var_types = dict(var_types_by_policy)
    for bcell in cells_cfg.get("branch_cells", []):
        all_var_types.setdefault(bcell["policy"], branch_var_types.get(bcell["id"]))

    output = {
        "meta": {
            "cells": [c["id"] for c in cells_cfg["cells"]],
            "branch_cells": [c["id"] for c in cells_cfg.get("branch_cells", [])],
            "eval_terminal_phase": "Outcomes", "post_target_phases": ["Impact"],
            "retriever_backend": eval_cfg["retriever"]["backend"],
            "embedding_threshold": eval_cfg["retriever"]["tau_grounding"],
            "threshold_calibration": eval_cfg.get("threshold_calibration", {}),
            "seed": eval_cfg["seed"],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "cell_design_correction": (
                "guideline's reference cell table names the third cell 'bk21_gemini'; "
                "actual data on disk is DeepSeek-generated (run_log.json). Cells here are "
                "100e_gemini/100e_deepseek/bk21_deepseek -- see configs/cells.yaml header."
            ),
            "load_reports": {cid: d["load_report"] for cid, d in by_cell.items()},
            "kg_tier_counts": {pid: k["tier_counts"] for pid, k in kg_meta.items()},
            "kg_meta": kg_meta,
            "variable_typing": {
                pid: {
                    "anchored": sum(1 for r in vt.values() if r.var_type == "anchored_constant"),
                    "predicted": sum(1 for r in vt.values() if r.var_type == "predicted"),
                    "unanchored": sum(1 for r in vt.values() if r.var_type == "unanchored_constant"),
                    "headcount_mislabeled": [f"{p}.{v}" for (p, v), r in vt.items()
                                              if r.var_type == "headcount_mislabeled"],
                }
                for pid, vt in all_var_types.items() if vt is not None
            },
        },
        "by_cell": by_cell,
        "by_branch_cell": by_branch_cell,
        "contrasts": contrasts,
        "branch_contrasts": branch_contrasts,
        "evidence": evidence_all,
        "influence_graphs": influence_graphs_all,
    }

    for cid, data in by_cell.items():
        data.pop("load_report", None)

    out_path = ROOT / "outputs" / "evaluation.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False, default=str)
    log(f"wrote {out_path}")
    return output


if __name__ == "__main__":
    run_pipeline()
