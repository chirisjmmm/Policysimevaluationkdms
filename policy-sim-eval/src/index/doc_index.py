"""Document paragraph-chunk index, independent of KG chunk_ids (sec 4)."""
from __future__ import annotations

from dataclasses import dataclass

from src.index.retriever import DEFAULT_MULTILINGUAL_MODEL, Retriever
from src.io.normalize import extract_quantities
from src.prep.doc_clean import DocCleanResult


@dataclass(frozen=True)
class DocChunk:
    id: str
    page: int | None
    text: str
    quantities: tuple
    raw_span: tuple


class DocIndex:
    def __init__(self, clean_result: DocCleanResult, retriever_backend: str = "tfidf",
                 embedding_model: str = DEFAULT_MULTILINGUAL_MODEL):
        self.clean_result = clean_result
        self.chunks: list[DocChunk] = self._build_chunks()
        self._retriever: Retriever | None = None
        self._retriever_backend = retriever_backend
        self._embedding_model = embedding_model

    def _build_chunks(self) -> list[DocChunk]:
        text = self.clean_result.clean_text
        chunks: list[DocChunk] = []
        cursor = 0
        idx = 0
        for para in text.split("\n\n"):
            if not para.strip():
                cursor += len(para) + 2
                continue
            start = text.index(para, cursor)
            end = start + len(para)
            cursor = end
            raw_span = self.clean_result.to_raw((start, end))
            page = self.clean_result.page_at_raw_offset(raw_span[0])
            qs = tuple(extract_quantities(para))
            chunks.append(DocChunk(f"doc_{idx:04d}", page, para, qs, raw_span))
            idx += 1
        return chunks

    def all_text(self) -> str:
        return self.clean_result.clean_text

    def retriever(self) -> Retriever:
        if self._retriever is None:
            self._retriever = Retriever([c.text for c in self.chunks], backend=self._retriever_backend,
                                          embedding_model=self._embedding_model)
        return self._retriever
