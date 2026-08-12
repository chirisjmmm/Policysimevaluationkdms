"""DM1 Interaction Structure Diversity, DM2 Coalition Variability (sec 5.5).

Data limitation: `revised_posts` is always a byte-for-byte no-op copy of
`initial_posts` in this dataset (verified at load time), so there is no
pre/post-influence pair to measure the shrinkage(j,i) component of the
influence-graph edge weight against. Edge weight is therefore computed from
only the two measurable components (value_uptake, name_citation) with
shrinkage fixed at 0 -- i.e. max edge weight is 0.7, not 1.0. This is a
structural property of the data, not a bug; the theta_disconnected /
theta_complete thresholds in config are relative to this same scale.
"""
from __future__ import annotations

import re
from collections import defaultdict

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import pdist
from scipy.stats import entropy
from sklearn.metrics import adjusted_rand_score

from src.io.ir import PHASE_ORDER

_NAME_STOPWORD_CLASH = {"an", "a", "i", "the", "in", "on", "at", "is", "he", "she",
                          "it", "we", "us", "to", "of", "so", "as", "by", "mr", "mr."}


def _name_citation(text: str, name: str) -> bool:
    if not name or len(name) < 3 or name.lower() in _NAME_STOPWORD_CLASH:
        return False
    pattern = re.compile(rf"\b{re.escape(name)}(?:'s)?\b", re.IGNORECASE)
    return bool(pattern.search(text))


def _value_uptake_pair(from_post, to_post) -> bool:
    from_values = {v for v in from_post.prediction_values.values() if isinstance(v, (int, float))}
    mine = {v for v in to_post.prediction_values.values() if isinstance(v, (int, float))}
    from src.io.normalize import extract_quantities
    seen = {q.value for q in extract_quantities(to_post.narrative or "")}
    return bool((seen & from_values) - mine)


def build_influence_graph(phase_state, w_uptake: float = 0.5, w_citation: float = 0.2) -> dict:
    posts = sorted(phase_state.posts, key=lambda p: p.order_index)
    edges = {}
    for i, pi in enumerate(posts):
        for j, pj in enumerate(posts):
            if j <= i:
                continue
            uptake = _value_uptake_pair(pi, pj)
            citation = _name_citation(pj.narrative or "", pi.agent_key)
            weight = w_uptake * uptake + w_citation * citation
            if weight > 0:
                edges[(pi.role_slot, pj.role_slot, i, j)] = weight
    return edges


def _classify_topology(edges: dict, n_agents: int, theta_disconnected: float, theta_complete: float) -> str:
    total_weight = sum(edges.values())
    n_possible = n_agents * (n_agents - 1) // 2
    if total_weight < theta_disconnected:
        return "disconnected"

    in_degree = defaultdict(int)
    n_existing = 0
    immediate_precedessor_hits = 0
    for (src_slot, dst_slot, i, j), w in edges.items():
        in_degree[dst_slot] += 1
        n_existing += 1
        if j == i + 1:
            immediate_precedessor_hits += 1

    if in_degree:
        max_in = max(in_degree.values())
        mean_in = np.mean(list(in_degree.values()))
        max_share = max_in / max(n_existing, 1)
        if mean_in > 0 and (max_in / mean_in) >= 2.0 and max_share >= 0.5:
            return "star"

    if n_existing > 0 and (immediate_precedessor_hits / n_existing) >= 0.7:
        return "chain"

    # guideline: "위 해당 없고 밀도 >= 0.5" -- complete is the catch-all 4th bucket
    # (entropy normalization assumes exactly 4 categories partition all runs).
    return "complete"


def run_dm1(runs: list, config: dict) -> dict:
    w_uptake = config.get("w_uptake", 0.5)
    w_citation = config.get("w_citation", 0.2)
    theta_disc = config.get("theta_disconnected", 0.15)
    theta_complete = config.get("theta_complete", 0.5)

    topology_counts = {"disconnected": 0, "star": 0, "chain": 0, "complete": 0}
    anchor_role_dist: dict = defaultdict(int)
    by_phase_counts = {p: {"disconnected": 0, "star": 0, "chain": 0, "complete": 0} for p in PHASE_ORDER}
    influence_graphs = {}

    for run in runs:
        for ph in run.phases:
            if not ph.posts:
                continue
            edges = build_influence_graph(ph, w_uptake, w_citation)
            topo = _classify_topology(edges, len(ph.posts), theta_disc, theta_complete)
            topology_counts[topo] += 1
            by_phase_counts[ph.phase][topo] += 1
            influence_graphs[f"{run.cell_id}::{run.run_id}::{ph.phase}"] = {
                "edges": {f"{a}->{b}": w for (a, b, i, j), w in edges.items()}, "topology": topo,
            }
            if topo == "star":
                in_deg = defaultdict(int)
                for (a, b, i, j) in edges:
                    in_deg[b] += 1
                if in_deg:
                    anchor_slot = max(in_deg, key=in_deg.get)
                    stakeholder = next((p.stakeholder_type for p in ph.posts if p.role_slot == anchor_slot), "unknown")
                    anchor_role_dist[stakeholder] += 1

    total = sum(topology_counts.values())
    if total == 0:
        return {"DM1": None, "topology_counts": topology_counts, "by_phase": by_phase_counts,
                "anchor_role_dist": dict(anchor_role_dist)}

    probs = [c / total for c in topology_counts.values() if c > 0]
    h = entropy(probs, base=np.e) if probs else 0.0
    dm1 = h / np.log(4)

    return {
        "DM1": float(dm1), "topology_counts": topology_counts, "by_phase": by_phase_counts,
        "anchor_role_dist": dict(anchor_role_dist), "influence_graphs": influence_graphs,
    }


# --------------------------------------------------------------------------
# DM2: Coalition Variability
# --------------------------------------------------------------------------

def run_dm2(runs: list, var_types: dict, seed: int = 20260810) -> dict:
    by_phase_ari = {}
    for phase in PHASE_ORDER:
        pred_vars = sorted({v for (p, v), r in var_types.items() if p == phase and r.var_type == "predicted"})
        if not pred_vars:
            continue

        partitions = []
        for run in runs:
            try:
                ph = run.phase(phase)
            except KeyError:
                continue
            posts = sorted(ph.posts, key=lambda p: p.order_index)
            if len(posts) != 5:
                continue
            vectors = np.array([[p.prediction_values.get(v, np.nan) for v in pred_vars] for p in posts])
            if np.isnan(vectors).any():
                continue
            std = vectors.std(axis=0)
            std[std == 0] = 1.0
            z = (vectors - vectors.mean(axis=0)) / std
            try:
                d = pdist(z, metric="euclidean")
                if np.allclose(d, 0):
                    labels = np.zeros(5, dtype=int)
                else:
                    Z = linkage(d, method="average")
                    labels = fcluster(Z, t=2, criterion="maxclust")
            except Exception:
                continue
            partitions.append(labels)

        if len(partitions) < 2:
            continue
        aris = []
        n = len(partitions)
        rng = np.random.default_rng(seed)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        if len(pairs) > 2000:
            idx = rng.choice(len(pairs), 2000, replace=False)
            pairs = [pairs[k] for k in idx]
        for i, j in pairs:
            aris.append(adjusted_rand_score(partitions[i], partitions[j]))
        by_phase_ari[phase] = {"mean_ARI": float(np.mean(aris)), "n_runs": len(partitions), "n_pairs": len(aris)}

    all_means = [d["mean_ARI"] for d in by_phase_ari.values()]
    dm2 = 1 - float(np.mean(all_means)) if all_means else None
    return {"DM2": dm2, "by_phase": by_phase_ari}
