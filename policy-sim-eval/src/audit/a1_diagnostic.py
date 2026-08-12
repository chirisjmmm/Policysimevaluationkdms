"""A1 manual-review protocol Step 1 (v4 guideline sec 6.1): per-fact
diagnostic table. Fully automatic -- tells a human WHERE to spend their 20
minutes of alias review (Step 2) and WHETHER a high contradiction rate looks
like scope-derivation/matching noise before they burn time labeling
(Step 3). Does not itself decide anything; a human reads this table.
"""
from __future__ import annotations


def build_diagnostic_table(a1_result: dict) -> list:
    """a1_result: the dict returned by acc_micro.run_a1()."""
    rows = []
    for fact_id, stats in a1_result["by_fact"].items():
        mention = stats["mentions"]
        unscored = stats["unscored"]
        with_qty = mention - unscored
        scored = stats["supported"] + stats["contradictions"]
        rows.append({
            "fact_id": fact_id, "mention": mention, "with_qty": with_qty,
            "consistent": stats["supported"], "contradiction": stats["contradictions"],
            "unscored": unscored,
            "contradiction_rate": (stats["contradictions"] / scored) if scored else None,
        })
    rows.sort(key=lambda r: -r["mention"])
    return rows


def diagnose_patterns(rows: list, high_contradiction_threshold: float = 0.5,
                       low_mention_threshold: int = 2, low_with_qty_ratio: float = 0.5) -> list:
    """Flags per v4 sec 6.1 '진단 규칙' table. One flag per (fact, pattern)."""
    flags = []
    for r in rows:
        if r["mention"] <= low_mention_threshold:
            flags.append({
                "fact_id": r["fact_id"], "pattern": "mention_near_zero",
                "diagnosis": (
                    f"mention={r['mention']} -- 별칭이 실제 발화 표현과 불일치할 가능성. "
                    "Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것."
                ),
            })
        elif r["mention"] > 0 and r["with_qty"] < r["mention"] * low_with_qty_ratio:
            flags.append({
                "fact_id": r["fact_id"], "pattern": "mention_high_with_qty_low",
                "diagnosis": (
                    f"mention={r['mention']}, with_qty={r['with_qty']} -- 언급은 하나 근처(±60자)에 "
                    "같은 단위 종류 수치가 없음. 별칭 보강으로는 안 고쳐짐; window 폭이나 단위필터 "
                    "로직 자체를 재검토할 사안."
                ),
            })
        if r["contradiction_rate"] is not None and r["contradiction_rate"] > high_contradiction_threshold:
            flags.append({
                "fact_id": r["fact_id"], "pattern": "high_contradiction_rate",
                "diagnosis": (
                    f"모순 비율 {r['contradiction_rate']:.0%} ({r['contradiction']}/"
                    f"{r['consistent'] + r['contradiction']}) -- 스코프 파생 또는 매칭 오류 의심. "
                    "Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것."
                ),
            })
    return flags


def format_markdown(rows: list, flags: list) -> str:
    lines = [
        "| fact_id | mention | with_qty | consistent | contradiction | unscored | contradiction_rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        cr = f"{r['contradiction_rate']:.0%}" if r["contradiction_rate"] is not None else "-"
        lines.append(f"| {r['fact_id']} | {r['mention']} | {r['with_qty']} | {r['consistent']} | "
                      f"{r['contradiction']} | {r['unscored']} | {cr} |")
    if flags:
        lines.append("")
        lines.append("**진단 플래그**")
        lines.append("")
        for f in flags:
            lines.append(f"- `{f['fact_id']}` [{f['pattern']}]: {f['diagnosis']}")
    return "\n".join(lines)


def run_diagnostic(a1_result: dict, cell_id: str, out_path=None) -> str:
    rows = build_diagnostic_table(a1_result)
    flags = diagnose_patterns(rows)
    md = f"## A1 수기 검토 Step 1 -- {cell_id} 사실별 진단표\n\n" + format_markdown(rows, flags)
    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(md)
    return md
