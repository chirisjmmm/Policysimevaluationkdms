"""Branch tree -> Run extraction, at two different sample-unit levels (v4
guideline sec 8.2), plus the n_eff correlation correction (sec 8.3).

Sample unit differs by metric family because branch Run objects share
ancestors and are therefore not independent:
  level="outcomes" -> 27 Outcomes-phase nodes -> X1, X4, DX1
  level="leaf"      -> 81 Impact-phase (leaf) nodes -> A1, A2, M1, M2, D1, D2, DM1
"""
from __future__ import annotations

import statistics
from collections import defaultdict

from src.branch.tree_loader import BranchTree
from src.io.ir import PHASE_INDEX, PHASE_ORDER, PhaseState, Post, Run
from src.io.loaders import _assign_role_slots

LEVEL_DEPTH = {"outcomes": 3, "leaf": 4}


def _root_participants(tree: BranchTree) -> list:
    """Persona roster is fixed across the whole tree (verified: single
    posting_order pattern across all 121 nodes in both NSA and K-pass) --
    read it once from the root node rather than requiring a separate
    participants list the branch schema doesn't provide."""
    root_posts = tree.nodes["root"]["phase_result"]["initial_posts"]
    return [{"name": p["persona_name"], "stakeholder_type": p["stakeholder_type"],
             "persona_id": p["persona_name"]} for p in root_posts]


def _node_to_phase_state(node_id: str, node: dict, cell_id: str, run_id: str,
                          name_to_slot: dict) -> PhaseState:
    pr = node["phase_result"]
    posts = []
    for order_index, post in enumerate(pr["initial_posts"]):
        name = post["persona_name"]
        slot = name_to_slot.get(name, f"{post.get('stakeholder_type', 'Unknown')}#1")
        posts.append(Post(
            cell_id=cell_id, run_id=run_id, path_key=node_id,
            phase=pr["phase"], phase_index=node["phase_index"], order_index=order_index,
            agent_key=name, stakeholder_type=post.get("stakeholder_type", "Unknown"),
            role_slot=slot, prediction_values=post.get("prediction_values") or {},
            narrative=post.get("narrative") or "", evidence=tuple(post.get("evidence") or []),
            transition_mode=node.get("transition_mode"),
        ))
    return PhaseState(
        pr["phase"], node["phase_index"], tuple(posts), pr.get("phase_summary") or "",
        tuple(pr.get("posting_order") or []), {},
    )


def extract_runs(tree: BranchTree, level: str) -> list:
    depth = LEVEL_DEPTH[level]
    outcomes_depth = LEVEL_DEPTH["outcomes"]
    participants = _root_participants(tree)
    name_to_slot = _assign_role_slots(participants)

    target_keys = [k for k, v in tree.nodes.items() if v["phase_index"] == depth]
    runs = []
    for key in sorted(target_keys):
        chain_ids = []
        cur = key
        while cur is not None:
            chain_ids.append(cur)
            cur = tree.nodes[cur]["parent_id"]
        chain_ids.reverse()  # root ... key

        phases = tuple(
            _node_to_phase_state(nid, tree.nodes[nid], tree.cell_id, key, name_to_slot)
            for nid in chain_ids
        )

        target_values = {}
        if depth >= outcomes_depth:
            oc_node_id = chain_ids[outcomes_depth]
            oc_posts = tree.nodes[oc_node_id]["phase_result"]["initial_posts"]
            var_names = {v for p in oc_posts for v in p["prediction_values"]}
            for var in var_names:
                vals = [p["prediction_values"][var] for p in oc_posts if var in p["prediction_values"]]
                if vals:
                    target_values[var] = statistics.mean(vals)

        transition_path = tuple(key.split("/")[1:]) if key != "root" else tuple()
        runs.append(Run(
            cell_id=tree.cell_id, run_id=key, source="branch", transition_path=transition_path,
            phases=phases, target_values=target_values, participants=tuple(participants),
            scenario_impact={},
        ))
    return runs


# --------------------------------------------------------------------------
# n_eff / ICC_subtree (sec 8.3) -- branch Outcomes-27 runs share ancestors
# (each depth-1 subtree of "enabling"/"baseline"/"constraining" covers 9 of
# the 27), so they are not independent observations. Uses the same
# population-SS-ratio ICC convention as div_micro.run_d1 (D1/ICC_role) for
# consistency within this codebase, rather than switching to classical
# mean-square ANOVA ICC partway through.
# --------------------------------------------------------------------------

def compute_n_eff(runs: list, var: str) -> dict:
    """runs must be the outcomes-level (27-node) extraction. Groups by the
    depth-1 ancestor (first element of transition_path) as the subtree key."""
    values_by_subtree = defaultdict(list)
    for run in runs:
        if var not in run.target_values or not run.transition_path:
            continue
        subtree_key = run.transition_path[0]
        values_by_subtree[subtree_key].append(run.target_values[var])

    all_values = [v for vs in values_by_subtree.values() for v in vs]
    n = len(all_values)
    if n < 2 or len(values_by_subtree) < 2:
        return {"ICC_subtree": None, "n_eff": float(n), "n": n, "m": None}

    grand_mean = statistics.mean(all_values)
    ss_total = sum((v - grand_mean) ** 2 for v in all_values)
    ss_between = sum(len(vs) * (statistics.mean(vs) - grand_mean) ** 2 for vs in values_by_subtree.values())
    icc_subtree = (ss_between / ss_total) if ss_total > 0 else 0.0
    m = statistics.mean(len(vs) for vs in values_by_subtree.values())
    n_eff = n / (1 + (m - 1) * icc_subtree)
    return {"ICC_subtree": float(icc_subtree), "n_eff": float(n_eff), "n": n, "m": float(m)}
