"""Policy document cleaning with an offset map back to raw text (sec 3.1).

Pipeline: strip page markers (keep page metadata) -> strip image placeholders
-> strip running header/footer noise -> drop ToC pages -> strip stray
footnote markers glued to numbers (BK21) -> normalize curly quotes ->
collapse whitespace. Every step preserves a per-character offset back to the
raw string so evidence spans can be reported in original document
coordinates.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from dataclasses import dataclass, field

RE_PAGE_MARKER = re.compile(r"\n?-{2,3}\s*page\s+(\d+)\s*-{2,3}\n?", re.IGNORECASE)
RE_IMAGE_PLACEHOLDER = re.compile(r"!\[image\s*\d*\]\([^)]*\)")
RE_HEADER_FOOTER = re.compile(r"^[가-힣A-Za-z0-9 .·]{1,30}\|\s*\d{1,4}\s*$", re.MULTILINE)
RE_FOOTNOTE_GLUE = re.compile(r"(?<=[가-힣])(\d{1,2})\)")
RE_TABLE_SEPARATOR_ROW = re.compile(r"^\|[\s:|-]+\|\s*$", re.MULTILINE)
RE_TOC_HEADING = re.compile(r"^\s*(contents|차\s*례|목\s*차)\s*$", re.IGNORECASE | re.MULTILINE)

_CURLY_QUOTES = {
    "‘": "'", "’": "'", "“": '"', "”": '"',
    "–": "-", "—": "-", " ": " ",
}


@dataclass
class DocCleanResult:
    clean_text: str
    offset_map: list[int]          # clean_text[i] came from raw_text[offset_map[i]]
    page_boundaries: list[tuple[int, int]] = field(default_factory=list)  # (raw_offset, page_num)
    stats: dict[str, int] = field(default_factory=dict)

    def to_raw(self, span: tuple[int, int]) -> tuple[int, int]:
        if not self.offset_map:
            return (0, 0)
        lo = self.offset_map[min(span[0], len(self.offset_map) - 1)]
        hi = self.offset_map[min(span[1] - 1, len(self.offset_map) - 1)] + 1
        return (lo, hi)

    def page_at_raw_offset(self, raw_offset: int) -> int | None:
        if not self.page_boundaries:
            return None
        offsets = [b[0] for b in self.page_boundaries]
        idx = bisect_right(offsets, raw_offset) - 1
        if idx < 0:
            return None
        return self.page_boundaries[idx][1]


class _T:
    """(text, offsets) pair threaded through cleaning passes."""

    __slots__ = ("text", "offsets")

    def __init__(self, text: str, offsets: list[int]):
        self.text = text
        self.offsets = offsets

    @classmethod
    def from_raw(cls, raw: str) -> "_T":
        return cls(raw, list(range(len(raw))))

    def remove(self, pattern: re.Pattern) -> tuple["_T", int]:
        chunks, offs, last, n = [], [], 0, 0
        for m in pattern.finditer(self.text):
            chunks.append(self.text[last:m.start()])
            offs.extend(self.offsets[last:m.start()])
            last = m.end()
            n += 1
        chunks.append(self.text[last:])
        offs.extend(self.offsets[last:])
        return _T("".join(chunks), offs), n

    def remove_span(self, start: int, end: int) -> "_T":
        return _T(self.text[:start] + self.text[end:], self.offsets[:start] + self.offsets[end:])

    def substitute_chars(self, table: dict[str, str]) -> "_T":
        return _T("".join(table.get(c, c) for c in self.text), list(self.offsets))

    def collapse_whitespace(self) -> "_T":
        chunks, offs, i, n = [], [], 0, len(self.text)
        while i < n:
            c = self.text[i]
            if c in (" ", "\t"):
                chunks.append(" ")
                offs.append(self.offsets[i])
                while i < n and self.text[i] in (" ", "\t"):
                    i += 1
            elif c == "\n":
                j = i
                while j < n and self.text[j] in ("\n", " ", "\t"):
                    j += 1
                chunks.append("\n\n")
                offs.append(self.offsets[i])
                offs.append(self.offsets[i])
                i = j
            else:
                chunks.append(c)
                offs.append(self.offsets[i])
                i += 1
        return _T("".join(chunks), offs)


def _extract_page_boundaries(raw: str) -> list[tuple[int, int]]:
    return [(m.start(), int(m.group(1))) for m in RE_PAGE_MARKER.finditer(raw)]


def _is_toc_like_page(page_text: str) -> bool:
    lines = [l.strip() for l in page_text.splitlines() if l.strip()]
    if not lines:
        return False
    if RE_TOC_HEADING.search(page_text):
        return True
    dot_leader = page_text.count("···") >= 1 or "......" in page_text
    short_or_numeric = sum(1 for l in lines if re.fullmatch(r"[\divxIVX]{1,4}", l) or len(l) <= 20)
    return dot_leader or (short_or_numeric / len(lines) >= 0.7 and len(lines) >= 3)


def _drop_toc_pages(raw: str, page_boundaries: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Return list of (start, end) raw spans covering a single *contiguous*
    leading ToC block, found within the first few pages only. Never scans
    or removes pages elsewhere in the document (would risk deleting real
    content, e.g. an appendix budget table with many short numeric lines)."""
    if not page_boundaries:
        return []
    bounds = page_boundaries + [(len(raw), None)]
    search_limit = min(len(page_boundaries), 8)

    start_idx = None
    for i in range(search_limit):
        page_text = raw[bounds[i][0]:bounds[i + 1][0]]
        if RE_TOC_HEADING.search(page_text) or "···" in page_text or "......" in page_text:
            start_idx = i
            break
    if start_idx is None:
        return []

    end_idx = start_idx + 1
    while end_idx < search_limit:
        page_text = raw[bounds[end_idx][0]:bounds[end_idx + 1][0]]
        if _is_toc_like_page(page_text):
            end_idx += 1
        else:
            break

    return [(bounds[start_idx][0], bounds[end_idx][0])]


def normalize_with_map(raw_text: str) -> DocCleanResult:
    stats: dict[str, int] = {}

    page_boundaries = _extract_page_boundaries(raw_text)
    toc_spans = _drop_toc_pages(raw_text, page_boundaries)
    stats["toc_pages_removed"] = len(toc_spans)

    t = _T.from_raw(raw_text)
    if toc_spans:
        # remove from raw-index space by rebuilding offsets, skipping dropped spans
        keep_mask = [True] * len(raw_text)
        for s, e in toc_spans:
            for i in range(s, min(e, len(raw_text))):
                keep_mask[i] = False
        chunks, offs = [], []
        for i, c in enumerate(raw_text):
            if keep_mask[i]:
                chunks.append(c)
                offs.append(i)
        t = _T("".join(chunks), offs)

    t, n = t.remove(RE_IMAGE_PLACEHOLDER)
    stats["image_placeholders_removed"] = n

    t, n = t.remove(RE_HEADER_FOOTER)
    stats["header_footer_lines_removed"] = n

    t, n = t.remove(RE_FOOTNOTE_GLUE)
    stats["footnote_markers_removed"] = n

    t, n = t.remove(RE_TABLE_SEPARATOR_ROW)
    stats["table_separator_rows_removed"] = n

    t = t.substitute_chars({"<br>": " "})  # placeholder; real <br> handled below
    # <br> is multi-char; handle via regex removal with space substitution
    br_pattern = re.compile(r"<br\s*/?>", re.IGNORECASE)
    t, n = t.remove(br_pattern)
    stats["br_tags_removed"] = n

    t, _ = t.remove(RE_PAGE_MARKER)  # drop markers now that page metadata is captured

    multi_space_before = sum(1 for i in range(len(t.text) - 1)
                              if t.text[i] == " " and t.text[i + 1] == " ")
    stats["multi_space_runs_before_collapse"] = multi_space_before

    t = t.substitute_chars(_CURLY_QUOTES)
    t = t.collapse_whitespace()

    clean_text = t.text.strip()
    lead_trim = len(t.text) - len(t.text.lstrip())
    offset_map = t.offsets[lead_trim: lead_trim + len(clean_text)]

    return DocCleanResult(
        clean_text=clean_text,
        offset_map=offset_map,
        page_boundaries=page_boundaries,
        stats=stats,
    )
