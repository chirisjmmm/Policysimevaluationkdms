"""Branch (assumption-conditioned) pathway tree loader (v4 guideline sec 8.1).

Validates the tree against the schema empirically confirmed for both NSA and
K-pass: 121 nodes (1+3+9+27+81), phase_index == node-key depth, revised_posts
a byte-for-byte no-op of initial_posts, refined_posts/judgment always null.
Required-field violations raise (would silently corrupt every downstream
metric); K-pass's extra fields (5 extra top-level keys, a per-node
`mechanism_anchor` that is null in all 121 nodes) are additive, not
breaking, so they are logged in `meta["extra_top_level_keys"]` rather than
treated as a validation failure -- confirmed via direct inspection of both
tree files before writing this loader, not assumed.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field

from src.io.ir import PHASE_ORDER

REQUIRED_TRANSITIONS = {"enabling", "baseline", "constraining"}
EXPECTED_NODE_COUNT = 121
_KNOWN_TOP_KEYS = {
    "format_version", "generation_mode", "source_run_dir", "scenario_index", "scenario_id",
    "simulation_target", "branch_inputs", "transitions", "phases", "nodes",
    "completed_parent_ids", "phase_interaction_mode",
}


@dataclass
class BranchTree:
    cell_id: str
    nodes: dict
    meta: dict
    duplicate_narratives: dict


def _detect_duplicate_narratives(nodes: dict) -> dict:
    seen = defaultdict(list)
    total = 0
    for node_id, node in nodes.items():
        for post in node["phase_result"]["initial_posts"]:
            total += 1
            seen[post["narrative"]].append((node_id, post["persona_name"]))
    duplicated = {k: v for k, v in seen.items() if len(v) > 1}
    return {
        "total": total, "unique": len(seen), "n_duplicate_groups": len(duplicated),
        "n_duplicated_posts": sum(len(v) for v in duplicated.values()),
    }


def load_branch_tree(path: str, cell_id: str) -> BranchTree:
    with open(path, encoding="utf-8") as f:
        d = json.load(f)

    errors = []
    if d.get("format_version") != 1:
        errors.append(f"format_version = {d.get('format_version')!r}, expected 1")
    if d.get("phases") != PHASE_ORDER:
        errors.append(f"phases = {d.get('phases')!r}, expected {PHASE_ORDER!r}")
    if set(d.get("transitions", [])) != REQUIRED_TRANSITIONS:
        errors.append(f"transitions = {d.get('transitions')!r}, expected {REQUIRED_TRANSITIONS!r}")
    if d.get("branch_inputs") is not False:
        errors.append(f"branch_inputs = {d.get('branch_inputs')!r}, expected False")

    nodes = d.get("nodes", {})
    if len(nodes) != EXPECTED_NODE_COUNT:
        errors.append(f"node count = {len(nodes)}, expected {EXPECTED_NODE_COUNT}")
    for node_id, node in nodes.items():
        if node_id.count("/") != node.get("phase_index"):
            errors.append(f"node {node_id!r}: phase_index={node.get('phase_index')} "
                           f"!= path depth {node_id.count('/')}")

    if errors:
        raise ValueError(
            f"branch tree schema validation FAILED for {path} ({len(errors)} error(s)) "
            f"-- aborting rather than silently producing corrupted downstream metrics:\n  "
            + "\n  ".join(errors)
        )

    revise_violations = []
    null_violations = []
    for node_id, node in nodes.items():
        pr = node["phase_result"]
        if pr.get("revised_posts") is not None and pr["initial_posts"] != pr["revised_posts"]:
            revise_violations.append(node_id)
        if pr.get("refined_posts") is not None:
            null_violations.append((node_id, "refined_posts"))
        for post in pr["initial_posts"]:
            if post.get("judgment") is not None:
                null_violations.append((node_id, "judgment"))

    extra_top_keys = sorted(set(d.keys()) - _KNOWN_TOP_KEYS)
    dup = _detect_duplicate_narratives(nodes)

    meta = {k: v for k, v in d.items() if k != "nodes"}
    meta["extra_top_level_keys"] = extra_top_keys
    meta["revise_noop_verified"] = len(revise_violations) == 0
    meta["revise_noop_violations"] = revise_violations
    meta["null_fields_verified"] = len(null_violations) == 0
    meta["null_field_violations"] = null_violations

    return BranchTree(cell_id=cell_id, nodes=nodes, meta=meta, duplicate_narratives=dup)
