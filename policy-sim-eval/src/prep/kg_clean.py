"""KG cleaning (sec 3.2): near-duplicate summary sentences, duplicate edges."""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])\n+")


def split_sentences(text: str) -> list[str]:
    if not text:
        return []
    return [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]


def dedup_near_duplicate_sentences(text: str, threshold: float = 0.75) -> tuple[str, int]:
    """Drop sentences that are near-duplicates (SequenceMatcher ratio > threshold)
    of an already-kept sentence. Returns (deduped_text, n_removed)."""
    sents = split_sentences(text)
    kept: list[str] = []
    removed = 0
    for s in sents:
        if any(SequenceMatcher(None, s, k).ratio() > threshold for k in kept):
            removed += 1
            continue
        kept.append(s)
    return " ".join(kept), removed


def dedup_edges(edges: list[dict]) -> tuple[list[dict], int]:
    seen = set()
    out = []
    removed = 0
    for e in edges:
        key = (e.get("source"), e.get("target"), e.get("type"), e.get("fact"))
        if key in seen:
            removed += 1
            continue
        seen.add(key)
        out.append(e)
    return out, removed
