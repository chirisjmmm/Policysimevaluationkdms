"""KG index: tier assignment (T1_explicit/T2_weak/T3_inferred), fact_table,
text_pool, role_types (sec 3.2, 4).

Reality check against the actual 100E graph.json: `attributes.evidence_alignment`
is present on 100% of edges and always "supported" (i.e. not a useful
discriminator), and is absent on every node. So instead of trusting that
field, tiers are derived empirically from whether each node/edge's
source_spans / source_text can actually be found in the cleaned policy
document -- literal substring (or near-verbatim, ratio>=0.90) match => T1,
top-1 similarity to the closest doc chunk >= align_confidence_t2_min => T2,
otherwise T3.

align_confidence_t2_min calibration: with the embedding backend, a naive
threshold of 0.5 (reasonable for TF-IDF) is too permissive. Measured via a
cross-policy null distribution (800 real narrative sentences queried against
the *other* policy's KG corpus -- guaranteed topically-adjacent-but-unrelated
text, same domain/register/language): median similarity 0.43, p95 0.557,
max 0.734. paraphrase-multilingual-MiniLM-L12-v2 has a much higher similarity
"noise floor" than TF-IDF (neural embedding anisotropy -- broadly on-topic
text clusters together regardless of whether it's specifically related).
Default here (0.55) is that p95, i.e. calibrated so only ~5% of *definitely*
unrelated cross-policy text would spuriously clear the bar. Still an
imperfect proxy: some genuinely-generic-but-same-policy text (e.g. a
persona's self-introduction) can score in the same 0.55-0.7 range as truly
fact-grounded text -- a resolving-power limit of this embedding model at the
sentence level, not fully fixable by threshold alone. Documented, not hidden.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from difflib import SequenceMatcher

import numpy as np

from src.index.doc_index import DocIndex
from src.io.normalize import Quantity, extract_quantities
from src.prep.kg_clean import dedup_edges, dedup_near_duplicate_sentences

LITERAL_MATCH_RATIO = 0.90
DEFAULT_ALIGN_CONFIDENCE_T2_MIN = 0.55


@dataclass(frozen=True)
class Fact:
    node_id: str
    node_name: str
    attr_name: str
    quantity: Quantity
    tier: str


@dataclass(frozen=True)
class TextRef:
    source_id: str
    tier: str
    text: str
    kind: str  # node_summary | edge_fact | edge_source_text


@dataclass
class KGIndex:
    fact_table: dict
    text_pool: list
    role_types: set
    node_tiers: dict
    edge_tiers: dict
    stats: dict


def _alignment(span_text: str, doc_index: DocIndex) -> tuple:
    """Returns (literal, best_confidence, tfidf_sim, embedding_sim). Both
    similarity backends are always computed and reported (guideline's
    "TF-IDF 결과를 항상 함께 보고" principle) even though the embedding
    backend (multilingual, when configured) decides the tier for
    cross-lingual sources like BK21 where TF-IDF cannot match at all."""
    if not span_text or not span_text.strip():
        return False, 0.0, 0.0, 0.0
    literal = span_text.strip() in doc_index.all_text()
    if not doc_index.chunks:
        return literal, 0.0, 0.0, 0.0

    retr = doc_index.retriever()
    idx_tfidf, sim_tfidf = retr.top1(span_text, backend="tfidf")
    if retr.backend == "embedding":
        idx_emb, sim_emb = retr.top1(span_text, backend="embedding")
    else:
        idx_emb, sim_emb = -1, 0.0

    if sim_emb >= sim_tfidf and idx_emb >= 0:
        best_idx, best_conf = idx_emb, sim_emb
    else:
        best_idx, best_conf = idx_tfidf, sim_tfidf

    if not literal and best_idx >= 0:
        ratio = SequenceMatcher(None, span_text, doc_index.chunks[best_idx].text).ratio()
        if ratio >= LITERAL_MATCH_RATIO:
            literal = True
    return literal, best_conf, sim_tfidf, sim_emb


def _tier(literal: bool, confidence: float, t2_min: float) -> str:
    if literal:
        return "T1_explicit"
    if confidence >= t2_min:
        return "T2_weak"
    return "T3_inferred"


def build_kg_index(raw_kg: dict, doc_index: DocIndex,
                    align_confidence_t2_min: float = DEFAULT_ALIGN_CONFIDENCE_T2_MIN) -> KGIndex:
    ontology = raw_kg.get("ontology", {})
    role_types = {et["name"] for et in ontology.get("entity_types", []) if et.get("persona_eligible")}

    nodes = raw_kg.get("nodes", [])
    edges, n_dup_edges = dedup_edges(raw_kg.get("edges", []))

    node_tiers: dict = {}
    edge_tiers: dict = {}
    text_pool: list = []
    fact_table: dict = {}

    literal_flags: list = []
    confidences: list = []
    tfidf_confidences: list = []
    embedding_confidences: list = []
    summary_dedup_removed = 0

    for n in nodes:
        deduped_summary, n_removed = dedup_near_duplicate_sentences(n.get("summary") or "")
        summary_dedup_removed += n_removed

        spans = n.get("source_spans") or []
        best_literal, best_conf = False, 0.0
        for sp in spans:
            lit, conf, conf_tfidf, conf_emb = _alignment(sp.get("text", ""), doc_index)
            literal_flags.append(lit)
            confidences.append(conf)
            tfidf_confidences.append(conf_tfidf)
            embedding_confidences.append(conf_emb)
            best_literal = best_literal or lit
            best_conf = max(best_conf, conf)
        tier = _tier(best_literal, best_conf, align_confidence_t2_min) if spans else "T3_inferred"
        node_tiers[n["id"]] = tier

        if deduped_summary:
            text_pool.append(TextRef(n["id"], tier, deduped_summary, "node_summary"))

        for attr_name, attr_val in (n.get("attributes") or {}).items():
            if not isinstance(attr_val, str):
                continue
            for q in extract_quantities(attr_val):
                fact_table.setdefault(f"{n['id']}::{attr_name}", []).append(
                    Fact(n["id"], n.get("name", n["id"]), attr_name, q, tier)
                )

    for e in edges:
        text = e.get("source_text") or e.get("fact") or ""
        lit, conf, conf_tfidf, conf_emb = _alignment(text, doc_index)
        literal_flags.append(lit)
        confidences.append(conf)
        tfidf_confidences.append(conf_tfidf)
        embedding_confidences.append(conf_emb)
        tier = _tier(lit, conf, align_confidence_t2_min) if text else "T3_inferred"
        edge_tiers[e["id"]] = tier

        fact_text = e.get("fact") or ""
        if fact_text:
            text_pool.append(TextRef(e["id"], tier, fact_text, "edge_fact"))
        if text and text != fact_text:
            text_pool.append(TextRef(e["id"], tier, text, "edge_source_text"))

    tier_counts = {"T1": 0, "T2": 0, "T3": 0}
    for t in list(node_tiers.values()) + list(edge_tiers.values()):
        tier_counts[t.split("_")[0]] += 1

    stats = {
        "n_nodes": len(nodes),
        "n_edges": len(edges),
        "n_duplicate_edges_removed": n_dup_edges,
        "n_near_duplicate_summary_sentences_removed": summary_dedup_removed,
        "kg_span_literal_match_rate": (sum(literal_flags) / len(literal_flags)) if literal_flags else 0.0,
        "chunk_alignment_confidence_median": float(np.median(confidences)) if confidences else 0.0,
        "chunk_alignment_confidence_median_tfidf": float(np.median(tfidf_confidences)) if tfidf_confidences else 0.0,
        "chunk_alignment_confidence_median_embedding": float(np.median(embedding_confidences)) if embedding_confidences else 0.0,
        "retriever_backend_used": doc_index.retriever().backend,
        "tier_counts": tier_counts,
    }

    return KGIndex(
        fact_table=fact_table, text_pool=text_pool, role_types=role_types,
        node_tiers=node_tiers, edge_tiers=edge_tiers, stats=stats,
    )
