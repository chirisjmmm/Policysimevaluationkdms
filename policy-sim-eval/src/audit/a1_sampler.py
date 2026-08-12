"""A1 manual-review protocol Step 3 (v4 guideline sec 6.3): draw a seeded
random sample of CONTRADICTION verdicts to CSV with an empty `label` column
for a human to fill in as A (true contradiction) / B (scope-derived
estimate) / C (matching error). This script only draws the sample and
writes the CSV -- it does not and cannot label anything itself; A1's
reported precision is not meaningful until a human fills the label column
and someone computes |A| / 100 from it.
"""
from __future__ import annotations

import csv
import random


LABEL_GUIDE = """라벨 정의 (label 칸에 A, B, C 중 하나를 채워주세요):
  A = 진짜 모순      : 문서 "up to S$330,000" <-> 발화 "capped at S$500,000"
  B = 스코프 파생    : 문서 "more than S$500M" <-> 발화가 그 값을 인지한 채
                       하위 범위(연차별/기관별 등)를 별도로 추정
                       예: "national commitment exceeds S$500M, but 2023 R&D
                       allocation is ~S$180M"
  C = 매칭 오류      : "예산" 별칭 근처의 과제 수(125)가 금액과 대조된 경우 등
"""


def sample_contradictions(evidence: list, n: int = 100, seed: int = 20260810) -> list:
    contradictions = [e for e in evidence if e.verdict == "CONTRADICTION"]
    rng = random.Random(seed)
    if len(contradictions) <= n:
        sample = list(contradictions)
    else:
        sample = rng.sample(contradictions, n)
    rng.shuffle(sample)
    return sample


def write_audit_csv(evidence: list, cell_id: str, out_path: str, n: int = 100, seed: int = 20260810) -> int:
    sample = sample_contradictions(evidence, n, seed)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["idx", "fact_id", "cell_id", "run_id", "phase", "agent", "context_text",
                    "doc_value", "doc_bound", "uttered_value", "label"])
        for i, e in enumerate(sample, start=1):
            w.writerow([
                i, e.fact_id, e.cell_id, e.run_id, e.phase, e.agent_key, e.context_text,
                e.reference.get("value"), e.reference.get("bound"),
                e.observed.value if e.observed else "", "",
            ])
    return len(sample)


def compute_precision(labeled_csv_path: str) -> dict:
    """Run after a human has filled the `label` column. Missing/blank labels
    are excluded from the denominator (not counted as any category) and
    reported separately so an incomplete review can't silently inflate a
    score with unlabeled rows."""
    counts = {"A": 0, "B": 0, "C": 0}
    unlabeled = 0
    total = 0
    with open(labeled_csv_path, encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            total += 1
            label = (row.get("label") or "").strip().upper()
            if label in counts:
                counts[label] += 1
            else:
                unlabeled += 1
    labeled = total - unlabeled
    precision = (counts["A"] / labeled) if labeled else None
    return {
        "n_sampled": total, "n_labeled": labeled, "n_unlabeled": unlabeled,
        "true_contradiction_A": counts["A"], "scope_derived_B": counts["B"], "match_error_C": counts["C"],
        "precision": precision,
    }
