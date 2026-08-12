"""A1 Factual Contradiction, A2 Grounding Profile (sec 5.1).

A1 is fact-anchored (loop over the curated fact_table first, then search
agent text for alias mentions) rather than post-anchored, so that aliases
not in a fixed dictionary don't silently disappear from recall -- the
guideline is explicit that starting from the utterance side loses recall.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from src.index.doc_index import DocIndex
from src.index.kg_index import KGIndex
from src.index.retriever import Retriever, batch_pairwise_similarity
from src.io.ir import PHASE_ORDER, Run
from src.io.normalize import Quantity, quantities_near, satisfies
from src.prep.kg_clean import split_sentences


# --------------------------------------------------------------------------
# A1: Factual Contradiction
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class A1Evidence:
    fact_id: str
    verdict: str            # SUPPORTED | CONTRADICTION
    cell_id: str
    run_id: str
    phase: str
    agent_key: str
    role_slot: str
    utterance_span: tuple
    utterance_text: str
    context_text: str        # +-200 chars around the quantity span, for audit CSV
    observed: Quantity
    reference: dict
    note: str = ""


def _fact_to_quantity(fact: dict) -> Quantity:
    return Quantity(
        value=float(fact["value"]), kind=fact["kind"], currency=fact.get("currency"),
        bound=fact.get("bound", "exact"), range_hi=fact.get("range_hi"),
        approx=bool(fact.get("approx", False)), raw=fact.get("evidence", ""), span=(0, 0),
    )


def _find_alias_mentions(text: str, aliases: list) -> list:
    spans = []
    for alias in aliases:
        if not alias:
            continue
        for m in re.finditer(re.escape(alias), text, re.IGNORECASE):
            spans.append((m.start(), m.end()))
    return spans


def run_a1(runs: list, fact_table: list) -> dict:
    """Returns {"supported":n,"contradicted":n,"unscored":n,"score":f,
    "by_fact":{...}, "evidence":[A1Evidence,...]}"""
    ref_quantities = {f["id"]: _fact_to_quantity(f) for f in fact_table}
    by_fact = {f["id"]: {"mentions": 0, "supported": 0, "contradictions": 0, "unscored": 0} for f in fact_table}
    evidence: list = []
    supported = contradicted = unscored = 0

    for run in runs:
        for post in run.all_posts():
            text = post.narrative or ""
            evidence_text = " ".join(post.evidence) if post.evidence else ""
            combined = text + " " + evidence_text
            for fact in fact_table:
                mentions = _find_alias_mentions(combined, fact.get("aliases", []))
                if not mentions:
                    continue
                by_fact[fact["id"]]["mentions"] += 1
                ref_q = ref_quantities[fact["id"]]
                found_quantity = False
                is_contradiction = False
                best_ev = None
                for m_start, m_end in mentions:
                    for q in quantities_near(combined, (m_start, m_end), window=60):
                        if q.kind != ref_q.kind:
                            continue
                        found_quantity = True
                        ok = satisfies(q, ref_q)
                        if not ok:
                            is_contradiction = True
                            best_ev = q
                        elif best_ev is None:
                            best_ev = q
                if not found_quantity:
                    unscored += 1
                    by_fact[fact["id"]]["unscored"] += 1
                    continue
                verdict = "CONTRADICTION" if is_contradiction else "SUPPORTED"
                if is_contradiction:
                    contradicted += 1
                    by_fact[fact["id"]]["contradictions"] += 1
                else:
                    supported += 1
                    by_fact[fact["id"]]["supported"] += 1
                span = best_ev.span if best_ev else (0, 0)
                ctx_lo, ctx_hi = max(0, span[0] - 200), min(len(combined), span[1] + 200)
                evidence.append(A1Evidence(
                    fact_id=fact["id"], verdict=verdict, cell_id=post.cell_id, run_id=post.run_id,
                    phase=post.phase, agent_key=post.agent_key, role_slot=post.role_slot,
                    utterance_span=span,
                    utterance_text=best_ev.raw if best_ev else "",
                    context_text=combined[ctx_lo:ctx_hi].strip(),
                    observed=best_ev, reference={
                        "id": fact["id"], "value": fact["value"], "kind": fact["kind"],
                        "bound": fact.get("bound", "exact"), "evidence": fact.get("evidence", ""),
                        "page": fact.get("page"),
                    },
                ))

    denom = supported + contradicted
    score = (supported / denom) if denom > 0 else None
    return {
        "supported": supported, "contradicted": contradicted, "unscored": unscored,
        "score": score, "by_fact": by_fact, "evidence": evidence,
    }


# --------------------------------------------------------------------------
# A2: Grounding Profile
# --------------------------------------------------------------------------

@dataclass
class A2Result:
    grounded_total: float           # the only score -- v4 dropped AnchorRetention/
    by_phase: dict                  # ProfileFit/TransitionScore (sec "삭제한 지표와 사유":
                                     # AnchorRetention penalizes Impact for correctly
                                     # relying on prior phases rather than the doc, which
                                     # is normal ToC structure not a defect; ProfileFit's
                                     # "expected" bands were arbitrary; TransitionScore is
                                     # a 5-point rank correlation with no real power).
                                     # by_phase (doc/kg/prior/none shares) is kept as
                                     # descriptive reporting, not folded into a score.


def run_a2(runs: list, doc_index: DocIndex, kg_index: KGIndex | None, config: dict) -> A2Result:
    tau = config.get("retriever", {}).get("tau_grounding", 0.35)
    grounding_tiers = set(config.get("kg", {}).get("use_tiers_for_grounding", ["T1_explicit", "T2_weak"]))
    backend = config.get("retriever", {}).get("backend", "tfidf")
    embedding_model = config.get("retriever", {}).get("embedding_model", "paraphrase-multilingual-MiniLM-L12-v2")

    doc_retr = doc_index.retriever()
    kg_corpus = []
    if kg_index is not None:
        kg_corpus = [t.text for t in kg_index.text_pool if t.tier in grounding_tiers]
    kg_retr = Retriever(kg_corpus, backend=backend, embedding_model=embedding_model)

    # Pass 1: flatten every narrative sentence out (with its phase + prior-text
    # context) so doc/kg similarity can be computed in ONE batched encode call
    # each, instead of one encode() per sentence -- the embedding backend's
    # per-call overhead makes the naive per-sentence loop ~4x+ slower at the
    # scale of thousands of sentences per cell (measured).
    flat_sentences: list = []
    flat_phase: list = []
    flat_prior_texts: list = []

    for run in runs:
        for ph_idx, phase_name in enumerate(PHASE_ORDER):
            try:
                ph = run.phase(phase_name)
            except KeyError:
                continue
            prev_phase_summary = None
            if ph_idx > 0:
                try:
                    prev_phase_summary = run.phase(PHASE_ORDER[ph_idx - 1]).phase_summary
                except KeyError:
                    prev_phase_summary = None
            ordered_posts = sorted(ph.posts, key=lambda p: p.order_index)
            for k, post in enumerate(ordered_posts):
                prior_texts = [p.narrative for p in ordered_posts[:k] if p.narrative]
                if prev_phase_summary:
                    prior_texts = prior_texts + [prev_phase_summary]
                for sent in split_sentences(post.narrative):
                    flat_sentences.append(sent)
                    flat_phase.append(phase_name)
                    flat_prior_texts.append(prior_texts)

    n = len(flat_sentences)
    doc_sims = doc_retr.top1_batch(flat_sentences) if doc_retr.corpus else [(-1, 0.0)] * n
    kg_sims = kg_retr.top1_batch(flat_sentences) if kg_retr.corpus else [(-1, 0.0)] * n

    # Prior pool MUST use the same similarity backend as doc/kg -- TF-IDF and
    # embedding cosine similarities are on different scales, so mixing them
    # in the same argmax biases toward whichever pool used the higher-scoring
    # method (found and fixed during calibration; see retriever.py docstring).
    prior_pairs = [(flat_sentences[i], " ".join(flat_prior_texts[i]) if flat_prior_texts[i] else "")
                   for i in range(n)]
    prior_sims = batch_pairwise_similarity(prior_pairs, backend=backend, embedding_model=embedding_model)

    phase_pool_counts = {p: {"doc": 0, "kg": 0, "prior": 0, "none": 0} for p in PHASE_ORDER}
    for i in range(n):
        sims = {"doc": doc_sims[i][1], "kg": kg_sims[i][1], "prior": prior_sims[i]}
        best_pool = max(sims, key=sims.get)
        pool = "none" if sims[best_pool] < tau else best_pool
        phase_pool_counts[flat_phase[i]][pool] += 1

    by_phase = {}
    none_shares = []
    for phase_name in PHASE_ORDER:
        counts = phase_pool_counts[phase_name]
        total = sum(counts.values())
        if total == 0:
            continue
        doc_share = counts["doc"] / total
        kg_share = counts["kg"] / total
        prior_share = counts["prior"] / total
        none_share = counts["none"] / total
        by_phase[phase_name] = {
            "doc_share": doc_share, "kg_share": kg_share,
            "prior_share": prior_share, "none_share": none_share, "n_sentences": total,
        }
        none_shares.append(none_share)

    grounded_total = 1 - (float(np.mean(none_shares)) if none_shares else 1.0)
    return A2Result(grounded_total, by_phase)
