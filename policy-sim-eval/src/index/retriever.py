"""TF-IDF cosine retriever with an optional multilingual sentence-embedding
backend, for grounding checks where the KG/agent text and the source
document are in different languages (BK21: KG + narratives are English,
source document is Korean -- TF-IDF is purely lexical and cannot bridge
that at all; verified empirically: cosine(Korean sentence, matching English
translation) ~0.70 vs ~0.04 for an unrelated pair, using
paraphrase-multilingual-MiniLM-L12-v2).

TF-IDF is always built regardless of backend, so callers can always report
both explicitly (`top1(query, backend="tfidf")` vs `backend="embedding"`)
per the guideline's "TF-IDF 결과를 항상 함께 보고" principle. The embedding
model is loaded once per process via a module-level cache (~7s cold load).
"""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DEFAULT_MULTILINGUAL_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

_EMBEDDER_CACHE: dict = {}


def _get_embedder(model_name: str):
    if model_name not in _EMBEDDER_CACHE:
        try:
            from sentence_transformers import SentenceTransformer
            _EMBEDDER_CACHE[model_name] = SentenceTransformer(model_name)
        except Exception:
            _EMBEDDER_CACHE[model_name] = None
    return _EMBEDDER_CACHE[model_name]


class Retriever:
    def __init__(self, corpus: list, backend: str = "tfidf",
                 embedding_model: str = DEFAULT_MULTILINGUAL_MODEL):
        self.corpus = corpus
        self.backend = "tfidf"
        self._embedder = None
        self._emb_matrix = None

        self._vectorizer = TfidfVectorizer()
        self._tfidf_matrix = self._vectorizer.fit_transform(corpus) if corpus else None

        if backend == "embedding":
            embedder = _get_embedder(embedding_model)
            if embedder is not None:
                self._embedder = embedder
                self.backend = "embedding"
                if corpus:
                    self._emb_matrix = embedder.encode(corpus, show_progress_bar=False)

    def top1(self, query: str, backend: str | None = None) -> tuple:
        """backend=None uses this retriever's primary backend. Pass "tfidf"
        or "embedding" explicitly to force one, for dual-reporting callers."""
        use = backend or self.backend
        if not self.corpus or not query.strip():
            return -1, 0.0
        if use == "embedding":
            if self._emb_matrix is None:
                return -1, 0.0
            q = self._embedder.encode([query], show_progress_bar=False)
            sims = cosine_similarity(q, self._emb_matrix)[0]
        else:
            q = self._vectorizer.transform([query])
            sims = cosine_similarity(q, self._tfidf_matrix)[0]
        idx = int(np.argmax(sims))
        return idx, float(sims[idx])

    def top1_batch(self, queries: list, backend: str | None = None) -> list:
        """Batched version of top1 -- encodes/vectorizes all queries in one
        call instead of one at a time. ~4x+ faster for embedding backend
        (measured: 200 individual calls 3.35s vs 1 batched call 0.9s), which
        matters at the scale of thousands of narrative sentences per cell."""
        use = backend or self.backend
        n = len(queries)
        if not self.corpus or n == 0:
            return [(-1, 0.0)] * n
        if use == "embedding":
            if self._emb_matrix is None:
                return [(-1, 0.0)] * n
            q = self._embedder.encode(list(queries), show_progress_bar=False)
            sims = cosine_similarity(q, self._emb_matrix)
        else:
            q = self._vectorizer.transform(queries)
            sims = cosine_similarity(q, self._tfidf_matrix)
        idxs = np.argmax(sims, axis=1)
        return [(int(idxs[i]), float(sims[i, idxs[i]])) for i in range(n)]

    def top_pool(self, query: str, pool_slices: dict) -> tuple:
        """argmax similarity across named contiguous slices of the corpus."""
        idx, sim = self.top1(query)
        if idx < 0:
            return "none", 0.0
        for name, (lo, hi) in pool_slices.items():
            if lo <= idx < hi:
                return name, sim
        return "none", sim


def batch_pairwise_similarity(pairs: list, backend: str = "tfidf",
                                embedding_model: str = DEFAULT_MULTILINGUAL_MODEL) -> list:
    """Batched (a,b) pairwise cosine similarity -- same backend-consistency
    requirement as pairwise_similarity, batched for the embedding backend's
    per-call overhead (see top1_batch)."""
    n = len(pairs)
    if n == 0:
        return []
    non_empty = [i for i, (a, b) in enumerate(pairs) if a.strip() and b.strip()]
    out = [0.0] * n
    if not non_empty:
        return out

    if backend == "embedding":
        embedder = _get_embedder(embedding_model)
        if embedder is not None:
            a_texts = [pairs[i][0] for i in non_empty]
            b_texts = [pairs[i][1] for i in non_empty]
            emb_a = embedder.encode(a_texts, show_progress_bar=False)
            emb_b = embedder.encode(b_texts, show_progress_bar=False)
            num = np.sum(emb_a * emb_b, axis=1)
            denom = np.linalg.norm(emb_a, axis=1) * np.linalg.norm(emb_b, axis=1) + 1e-9
            sims = num / denom
            for j, i in enumerate(non_empty):
                out[i] = float(sims[j])
            return out

    for i in non_empty:
        out[i] = pairwise_similarity(pairs[i][0], pairs[i][1], backend="tfidf")
    return out


def pairwise_similarity(a: str, b: str, backend: str = "tfidf",
                          embedding_model: str = DEFAULT_MULTILINGUAL_MODEL) -> float:
    """backend="embedding" MUST be used whenever this is compared against
    other embedding-backend similarity scores in the same argmax (e.g. A2's
    doc/kg/prior pool assignment) -- TF-IDF and embedding cosine similarities
    are not on the same scale, so mixing them biases the argmax toward
    whichever pool happens to use the higher-scoring method."""
    if not a.strip() or not b.strip():
        return 0.0
    if backend == "embedding":
        embedder = _get_embedder(embedding_model)
        if embedder is not None:
            emb = embedder.encode([a, b], show_progress_bar=False)
            return float(cosine_similarity([emb[0]], [emb[1]])[0][0])
    vec = TfidfVectorizer().fit([a, b])
    m = vec.transform([a, b])
    return float(cosine_similarity(m[0], m[1])[0][0])
