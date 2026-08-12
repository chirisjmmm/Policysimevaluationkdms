"""Boundary-aware numeric normalization (EVAL_GUIDELINE.md sec 3.5).

Parses quantities out of free text with explicit bound semantics
(exact / at_most / at_least / range) so that phrases like "up to S$330,000"
or "more than S$500 million" are not compared as point values.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

CURRENCY = {
    "S$": "SGD", "SGD": "SGD",
    "US$": "USD", "USD": "USD", "$": "USD",
    "₩": "KRW", "원": "KRW", "KRW": "KRW",
}

SCALE_EN = {
    "thousand": 1e3, "k": 1e3,
    "million": 1e6, "m": 1e6,
    "billion": 1e9, "bn": 1e9,
}

# BK21 (Korean) scale words. Ordered longest-match-first when compiled.
SCALE_KO = {
    "천": 1e3,        # 천
    "만": 1e4,        # 만
    "백만": 1e6,  # 백만
    "억": 1e8,        # 억
    "조": 1e12,       # 조
}

AT_MOST_PHRASES = [
    "up to", "not more than", "no more than", "at most", "no greater than",
    "최대", "이하", "미만",
]
AT_LEAST_PHRASES = [
    "more than", "over", "at least", "greater than", "exceeding",
    "committed more than",
    "이상", "초과",
]
APPROX_PHRASES = [
    "about", "approximately", "roughly", "around", "typically", "circa",
    "약", "대략", "정도",
]

_NUM = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_KO_SCALE_ALT = "|".join(sorted(SCALE_KO, key=len, reverse=True))
_EN_SCALE_ALT = "|".join(sorted(SCALE_EN, key=len, reverse=True))

RE_RANGE = re.compile(
    rf"(?P<lo>{_NUM})\s*(?:-|~|–|to)\s*(?P<hi>{_NUM})\s*"
    rf"(?P<unit>years?|months?|hours?|days?|weeks?|percent|%|"
    rf"년|개월|시간)",
    re.IGNORECASE,
)
RE_PCT_POINT = re.compile(
    rf"(?P<num>{_NUM})\s*(?:percentage\s+points?|pp\b|%\s*p\b|"
    rf"퍼센트\s*포인트|%\s*포인트)",
    re.IGNORECASE,
)
RE_PCT = re.compile(rf"(?P<num>{_NUM})\s*(?:%|percent|퍼센트)", re.IGNORECASE)
RE_CURRENCY_PREFIX = re.compile(
    rf"(?P<sym>S\$|US\$|\$|₩)\s*(?P<num>{_NUM})"
    rf"(?:\s*(?P<scale>{_EN_SCALE_ALT})\b)?",
    re.IGNORECASE,
)
RE_CURRENCY_KRW_SUFFIX = re.compile(
    rf"(?P<num>{_NUM})\s*(?:(?P<scale>{_KO_SCALE_ALT}))?\s*(?:원|KRW)\b",
)
RE_CURRENCY_EN_SUFFIX = re.compile(
    rf"(?P<num>{_NUM})\s*(?P<scale>{_EN_SCALE_ALT})\b\s*(?P<sym>SGD|USD|dollars?)",
    re.IGNORECASE,
)
RE_YEARS = re.compile(rf"(?P<num>{_NUM})\s*(?:years?|년간|개년|년)\b")
RE_MONTHS = re.compile(rf"(?P<num>{_NUM})\s*(?:months?|개월)\b", re.IGNORECASE)
RE_HOURS = re.compile(rf"(?P<num>{_NUM})\s*(?:hours?|시간)\b", re.IGNORECASE)
# "calendar"/"business" sit *between* the number and "days" (e.g. "30 calendar
# days", "60 business days" -- both common in US federal regulatory text like
# the No Surprises Act), so the qualifier is an optional infix, not a suffix.
# English "days" only -- deliberately NOT matching Korean "일", since that
# character is also the day-of-month marker ("5월 1일" = "May 1st") and would
# misparse ordinary calendar dates as a duration.
RE_DAYS = re.compile(rf"(?P<num>{_NUM})\s*(?:calendar\s+|business\s+)?days?\b", re.IGNORECASE)
RE_RANK = re.compile(rf"(?:rank(?:ed)?\s*(?:of|#)?\s*)(?P<num>{_NUM})", re.IGNORECASE)
RE_GENERIC = re.compile(rf"(?P<num>{_NUM})")

_APPROX_RE = re.compile("|".join(re.escape(p) for p in APPROX_PHRASES), re.IGNORECASE)
_AT_MOST_RE = re.compile("|".join(re.escape(p) for p in AT_MOST_PHRASES), re.IGNORECASE)
_AT_LEAST_RE = re.compile("|".join(re.escape(p) for p in AT_LEAST_PHRASES), re.IGNORECASE)


@dataclass(frozen=True)
class Quantity:
    value: float
    kind: str                 # currency|count|percent|percentage_point|rate|rank|hours|years|months
    currency: Optional[str]
    bound: str                 # exact | at_most | at_least | range
    range_hi: Optional[float]
    approx: bool
    raw: str
    span: tuple[int, int]


def _parse_num(s: str) -> float:
    return float(s.replace(",", ""))


def _bound_and_approx(text: str, start: int, end: int) -> tuple[str, bool]:
    before = text[max(0, start - 30):start]
    after = text[end:end + 15]
    bound = "exact"
    if _AT_MOST_RE.search(before) or _AT_MOST_RE.search(after):
        bound = "at_most"
    elif _AT_LEAST_RE.search(before) or _AT_LEAST_RE.search(after):
        bound = "at_least"
    approx = bool(_APPROX_RE.search(before))
    return bound, approx


class _Consumed:
    """Tracks already-matched character spans across extraction passes."""

    def __init__(self) -> None:
        self._spans: list[tuple[int, int]] = []

    def overlaps(self, start: int, end: int) -> bool:
        return any(not (end <= s or start >= e) for s, e in self._spans)

    def add(self, start: int, end: int) -> None:
        self._spans.append((start, end))


def extract_quantities(text: str) -> list[Quantity]:
    """Extract all quantities from `text`, most-specific pattern first."""
    out: list[Quantity] = []
    consumed = _Consumed()

    def emit(q: Quantity) -> None:
        out.append(q)
        consumed.add(q.span[0], q.span[1])

    # 1. Ranges ("typically 6-12 months")
    for m in RE_RANGE.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        unit = m.group("unit").lower()
        if unit in ("%", "percent"):
            kind = "percent"
        elif unit.startswith("year") or unit == "년":
            kind = "years"
        elif unit.startswith("month") or unit == "개월":
            kind = "months"
        elif unit.startswith("hour") or unit == "시간":
            kind = "hours"
        elif unit.startswith("day"):
            kind = "days"
        elif unit.startswith("week"):
            kind = "weeks"
        else:
            kind = unit
        _, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(_parse_num(m.group("lo")), kind, None, "range",
                       _parse_num(m.group("hi")), approx, m.group(0), m.span()))

    # 2. Percentage points (must precede plain percent)
    for m in RE_PCT_POINT.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        bound, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(_parse_num(m.group("num")), "percentage_point", None,
                       bound, None, approx, m.group(0), m.span()))

    # 3. Percent
    for m in RE_PCT.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        bound, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(_parse_num(m.group("num")), "percent", None,
                       bound, None, approx, m.group(0), m.span()))

    # 4. Currency: prefix symbol form (S$330,000 / S$0.33 million)
    for m in RE_CURRENCY_PREFIX.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        sym = m.group("sym")
        currency = CURRENCY.get(sym, CURRENCY.get(sym.upper()))
        scale = SCALE_EN.get((m.group("scale") or "").lower(), 1.0)
        value = _parse_num(m.group("num")) * scale
        bound, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(value, "currency", currency, bound, None, approx,
                       m.group(0), m.span()))

    # 5. Currency: Korean won suffix (2.6억원 / 260,000,000원)
    for m in RE_CURRENCY_KRW_SUFFIX.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        scale = SCALE_KO.get(m.group("scale") or "", 1.0)
        value = _parse_num(m.group("num")) * scale
        bound, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(value, "currency", "KRW", bound, None, approx,
                       m.group(0), m.span()))

    # 6. Currency: English scale + SGD/USD suffix
    for m in RE_CURRENCY_EN_SUFFIX.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        scale = SCALE_EN.get(m.group("scale").lower(), 1.0)
        value = _parse_num(m.group("num")) * scale
        sym = m.group("sym").upper()
        currency = "USD" if sym.startswith("DOLLAR") else sym
        bound, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(value, "currency", currency, bound, None, approx,
                       m.group(0), m.span()))

    # 7. Years / Months / Hours / Days
    for regex, kind in ((RE_YEARS, "years"), (RE_MONTHS, "months"), (RE_HOURS, "hours"), (RE_DAYS, "days")):
        for m in regex.finditer(text):
            if consumed.overlaps(m.start(), m.end()):
                continue
            bound, approx = _bound_and_approx(text, m.start(), m.end())
            emit(Quantity(_parse_num(m.group("num")), kind, None, bound, None,
                           approx, m.group(0), m.span()))

    # 8. Rank
    for m in RE_RANK.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        bound, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(_parse_num(m.group("num")), "rank", None, bound, None,
                       approx, m.group(0), m.span()))

    # 9. Generic fallback -> count
    for m in RE_GENERIC.finditer(text):
        if consumed.overlaps(m.start(), m.end()):
            continue
        bound, approx = _bound_and_approx(text, m.start(), m.end())
        emit(Quantity(_parse_num(m.group("num")), "count", None, bound, None,
                       approx, m.group(0), m.span()))

    out.sort(key=lambda q: q.span[0])
    return out


def quantities_near(text: str, span: tuple[int, int], window: int = 60) -> list[Quantity]:
    lo, hi = max(0, span[0] - window), min(len(text), span[1] + window)
    local = extract_quantities(text[lo:hi])
    return [
        Quantity(q.value, q.kind, q.currency, q.bound, q.range_hi, q.approx,
                  q.raw, (q.span[0] + lo, q.span[1] + lo))
        for q in local
    ]


def satisfies(observed: Quantity, ref: Quantity, rel_tol: float = 0.005) -> bool:
    if observed.kind != ref.kind:
        return False
    if observed.kind == "currency" and observed.currency != ref.currency:
        return False
    tol = rel_tol * max(abs(observed.value), abs(ref.value), 1.0)
    if observed.approx or ref.approx:
        tol *= 4
    if ref.bound == "exact":
        return abs(observed.value - ref.value) <= tol
    if ref.bound == "at_most":
        return observed.value <= ref.value + tol
    if ref.bound == "at_least":
        return observed.value >= ref.value - tol
    if ref.bound == "range":
        return (ref.value - tol) <= observed.value <= (ref.range_hi + tol)
    raise ValueError(f"unknown bound: {ref.bound}")
