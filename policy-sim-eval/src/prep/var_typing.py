"""Variable-type classification (sec 3.4) -- the precondition every
Diversity metric depends on. `anchored_constant` variance is measurement
error; `predicted` variance is the diversity signal; `unanchored_constant`
is descriptive-only (zero variance with no textual anchor -- often a
"consensus hallucination": every agent agrees on a number the policy never
stated); `headcount_mislabeled` (v4 sec 3.4) is a variable whose name
implies a rate/percentage but whose observed values are headcounts (found
in BK21 -- see configs/policy_bk21.yaml). Every metric that filters on
`var_type == "predicted"` (M1, M2, D1, D2, DX1, DX2, DX3) automatically
excludes both `unanchored_constant` and `headcount_mislabeled` variables by
construction -- there is no separate exclusion list to keep in sync.

Resolution order, per guideline sec 3.4:
  1. config `variable_anchors` manual declaration (authoritative) -- this is
     also the only place `headcount_mislabeled` is ever assigned; nothing
     about the (phase, var) name or value distribution alone reliably
     implies mislabeling, so it is never auto-detected, only human-confirmed.
  2. auto alias search against doc_index / kg_index T1 facts
  3. zero variance + no anchor found -> flag `unanchored_constant`
  4. otherwise -> `predicted`
"""
from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field

from src.index.doc_index import DocIndex
from src.index.kg_index import KGIndex
from src.io.normalize import Quantity, satisfies

_STOPWORDS = {"of", "per", "the", "a", "an", "in", "for", "to", "and"}


@dataclass(frozen=True)
class VarTypeResult:
    var_type: str                    # anchored_constant | predicted | unanchored_constant
    source: str                      # "config" | "auto_doc" | "auto_kg" | "variance_heuristic"
    ref: Quantity | None = None
    evidence: str | None = None


def _var_name_words(var: str) -> list[str]:
    words = re.split(r"[_\s]+", var.lower())
    return [w for w in words if w and w not in _STOPWORDS]


def _auto_search(var: str, doc_index: DocIndex, kg_index: KGIndex | None) -> VarTypeResult | None:
    words = set(_var_name_words(var))
    if not words:
        return None
    best_chunk = None
    best_overlap = 0.0
    for chunk in doc_index.chunks:
        if not chunk.quantities:
            continue
        text_lower = chunk.text.lower()
        hit = sum(1 for w in words if w in text_lower)
        overlap = hit / len(words)
        if overlap > best_overlap:
            best_overlap = overlap
            best_chunk = chunk
    if best_chunk is not None and best_overlap >= 0.6:
        q = best_chunk.quantities[0]
        return VarTypeResult("anchored_constant", "auto_doc", q, best_chunk.text[:200])

    if kg_index is not None:
        for key, facts in kg_index.fact_table.items():
            node_id = key.split("::")[0]
            attr = key.split("::")[-1]
            attr_words = set(_var_name_words(attr)) | set(_var_name_words(node_id.split(":")[-1]))
            overlap = len(words & attr_words) / max(len(words), 1)
            for f in facts:
                if f.tier == "T1_explicit" and overlap >= 0.6:
                    return VarTypeResult("anchored_constant", "auto_kg", f.quantity, key)
    return None


def classify_variable(
    phase: str,
    var: str,
    values: list[float],
    config: dict,
    doc_index: DocIndex,
    kg_index: KGIndex | None = None,
) -> VarTypeResult:
    anchors = config.get("variable_anchors", {}) or {}
    key = f"{phase}.{var}"
    if key in anchors:
        decl = anchors[key]
        ref = None
        if decl.get("ref"):
            r = decl["ref"]
            ref = Quantity(
                value=float(r["value"]), kind=r["kind"], currency=r.get("currency"),
                bound=r.get("bound", "exact"), range_hi=r.get("range_hi"),
                approx=False, raw=decl.get("evidence", ""), span=(0, 0),
            )
        return VarTypeResult(decl["type"], "config", ref, decl.get("evidence"))

    auto = _auto_search(var, doc_index, kg_index)
    if auto is not None:
        return auto

    if not values:
        return VarTypeResult("predicted", "variance_heuristic")
    variance = statistics.pvariance(values) if len(values) > 1 else 0.0
    if variance == 0.0:
        return VarTypeResult("unanchored_constant", "variance_heuristic")
    return VarTypeResult("predicted", "variance_heuristic")


def classify_all_variables(
    runs: list,
    config: dict,
    doc_index: DocIndex,
    kg_index: KGIndex | None = None,
) -> dict:
    """Pools values across all given runs (typically all runs of one policy,
    across cells) per (phase, var). Returns {(phase, var): VarTypeResult}."""
    values_by_key: dict[tuple, list] = {}
    for run in runs:
        for ph in run.phases:
            for post in ph.posts:
                for var, val in post.prediction_values.items():
                    if isinstance(val, (int, float)):
                        values_by_key.setdefault((ph.phase, var), []).append(float(val))

    out = {}
    for (phase, var), values in values_by_key.items():
        out[(phase, var)] = classify_variable(phase, var, values, config, doc_index, kg_index)
    return out
