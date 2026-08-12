"""Load raw scenario_*.json files into the common IR (sec 2.1, 2.4, 3.3).

Verifies the invariants the guideline documents as "실측" (empirically
observed) rather than trusting them blindly: revised_posts is a no-op copy
of initial_posts, and cross_checks.aggregated_value equals the arithmetic
mean of the five agents' prediction_values.
"""
from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass, field

from src.io.ir import PHASE_INDEX, PHASE_ORDER, PhaseState, Post, Run


@dataclass
class LoadReport:
    n_runs: int = 0
    revise_noop_verified: bool = True
    revise_noop_violations: list = field(default_factory=list)
    aggregated_value_matches_mean: bool = True
    aggregated_value_mismatches: list = field(default_factory=list)
    null_field_violations: list = field(default_factory=list)   # backward_pass/judgment/refined_posts not null
    role_slot_collisions: list = field(default_factory=list)


def _assign_role_slots(participants: list[dict]) -> dict[str, str]:
    """Map persona_name -> role_slot ('AIResearcher#1'), disambiguating
    duplicate stakeholder_type. Falls back to persona_id if names collide."""
    counts: dict[str, int] = {}
    name_to_slot: dict[str, str] = {}
    seen_names: dict[str, int] = {}
    for p in participants:
        name = p.get("name", "")
        seen_names[name] = seen_names.get(name, 0) + 1
    for p in participants:
        st = p.get("stakeholder_type", "Unknown")
        counts[st] = counts.get(st, 0) + 1
        slot = f"{st}#{counts[st]}"
        name = p.get("name", "")
        if seen_names.get(name, 0) > 1:
            # ambiguous persona name (e.g. "An", "A") -> disambiguate by persona_id suffix
            key = f"{name}::{p.get('persona_id', '')}"
        else:
            key = name
        name_to_slot[key] = slot
    return name_to_slot


def _lookup_slot(name_to_slot: dict[str, str], name: str, participants: list[dict]) -> str:
    if name in name_to_slot:
        return name_to_slot[name]
    # ambiguous-name case: only one persona with this name among participants -> unreachable normally
    matches = [p for p in participants if p.get("name") == name]
    if len(matches) == 1:
        st = matches[0].get("stakeholder_type", "Unknown")
        return f"{st}#1"
    return f"Unknown#{name}"


def load_scenario_file(path: str, cell_id: str, report: LoadReport) -> Run:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    run_id = d["scenario_id"]
    participants = d.get("participants", [])
    name_to_slot = _assign_role_slots(participants)

    for null_field in ("backward_pass", "fwd_bwd_fwd_pass", "fwd_bwd_fwd_bwd_pass"):
        if d.get(null_field) is not None:
            report.null_field_violations.append((run_id, null_field))

    cross_checks_by_phase: dict[str, dict] = {}
    for cc in d.get("cross_checks") or []:
        cross_checks_by_phase.setdefault(cc["phase"], {})[cc["variable"]] = cc

    phases: list[PhaseState] = []
    for phase_name in PHASE_ORDER:
        phase_entry = next((p for p in d.get("forward_pass") or [] if p["phase"] == phase_name), None)
        if phase_entry is None:
            phases.append(PhaseState(phase_name, PHASE_INDEX[phase_name], tuple(), "", tuple(), {}))
            continue

        initial = phase_entry.get("initial_posts") or []
        revised = phase_entry.get("revised_posts")
        if revised is not None and revised != initial:
            report.revise_noop_verified = False
            report.revise_noop_violations.append((run_id, phase_name))
        if phase_entry.get("refined_posts") is not None:
            report.null_field_violations.append((run_id, f"{phase_name}.refined_posts"))

        posting_order = tuple(phase_entry.get("posting_order") or [p["persona_name"] for p in initial])

        posts = []
        for order_index, post in enumerate(initial):
            name = post["persona_name"]
            posts.append(Post(
                cell_id=cell_id, run_id=run_id, path_key=run_id,
                phase=phase_name, phase_index=PHASE_INDEX[phase_name], order_index=order_index,
                agent_key=name, stakeholder_type=post.get("stakeholder_type", "Unknown"),
                role_slot=_lookup_slot(name_to_slot, name, participants),
                prediction_values=post.get("prediction_values") or {},
                narrative=post.get("narrative") or "",
                evidence=tuple(post.get("evidence") or []),
                transition_mode=None,
            ))

        phase_cc = cross_checks_by_phase.get(phase_name, {})
        for var, cc in phase_cc.items():
            vals = [p.prediction_values[var] for p in posts if var in p.prediction_values]
            agg = cc.get("aggregated_value")
            if vals and agg is not None:
                mean = sum(vals) / len(vals)
                tol = 1e-6 * max(abs(mean), 1.0)
                if abs(mean - agg) > tol:
                    report.aggregated_value_matches_mean = False
                    report.aggregated_value_mismatches.append((run_id, phase_name, var, mean, agg))

        phases.append(PhaseState(phase_name, PHASE_INDEX[phase_name], tuple(posts),
                                   phase_entry.get("phase_summary") or "", posting_order, phase_cc))

    target_values = {
        var: cc["aggregated_value"]
        for var, cc in cross_checks_by_phase.get("Outcomes", {}).items()
        if cc.get("aggregated_value") is not None
    }

    report.n_runs += 1
    return Run(
        cell_id=cell_id, run_id=run_id, source="single", transition_path=tuple(),
        phases=tuple(phases), target_values=target_values,
        participants=tuple(participants), scenario_impact=d.get("scenario_impact") or {},
    )


def load_cell(runs_dir: str, cell_id: str, report: LoadReport | None = None) -> tuple[list[Run], LoadReport]:
    if report is None:
        report = LoadReport()
    paths = sorted(
        glob.glob(os.path.join(runs_dir, "scenario_*.json")),
        key=lambda p: int(os.path.splitext(os.path.basename(p))[0].split("_")[1]),
    )
    runs = [load_scenario_file(p, cell_id, report) for p in paths]
    return runs, report
