"""Reads outputs/evaluation.json and writes a human-readable summary
(outputs/report.md). Static text/tables only (v4 guideline sec 10, viz kept
minimal). Matches v4's flat by_cell schema (10 scored metrics + 3 gates) and
the by_branch_cell / B1 / n_eff structure for NSA and K-pass.
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent


def fmt(x, nd=3):
    if x is None:
        return "N/A"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _x1_table(x1: dict) -> list:
    lines = ["| 타깃 | true_value | source | APE mean | APE median | 95% CI |",
             "|---|---|---|---|---|---|"]
    for var, x in x1.items():
        if x.get("status") != "OK":
            lines.append(f"| {var} | - | {x.get('source','-')} | {x.get('status')} | - | - |")
        else:
            ci = x["CI95"]
            lines.append(f"| {var} | {x['true_value']} | {x.get('source','-')} | {fmt(x['APE_mean'],1)}% | "
                          f"{fmt(x['APE_median'],1)}% | [{fmt(ci[0],1)}, {fmt(ci[1],1)}]% |")
    return lines


def _x4_lines(x4: dict) -> list:
    lines = []
    for var, x in x4.items():
        if x.get("status") != "OK":
            lines.append(f"  - `{var}`: {x.get('status')}")
        else:
            lines.append(f"  - `{var}`: coverage_50={x['coverage_50']}, coverage_90={x['coverage_90']}, "
                          f"PIT={fmt(x['PIT'])}, CRPS={fmt(x['CRPS'],1)}")
    return lines


def _gates_line(gates: dict) -> str:
    g1, g2, g3 = gates["G1"], gates["G2"], gates["G3"]

    def onep(g, label):
        if g["pass"] is None:
            return f"{label} 해당없음"
        status = "통과" if g["pass"] else "실패"
        return f"{label} {status}(위반 {g['n_violations']}/{g['n_evaluated']})"

    g3_txt = f"완료율 {g3['completion_rate']:.1%}" if g3.get("completion_rate") is not None else "완료율 N/A"
    return f"{onep(g1,'체인 일관성')} · {onep(g2,'값 범위')} · {g3_txt}"


def _cell_section(a, cid: str, c: dict, is_branch: bool = False) -> None:
    a(f"### {cid}")
    a("")
    if is_branch:
        a(f"표본 단위: A2/M1/M2/D1/D2/DM1 = leaf {c['n_leaves']}개, X1/X4/DX1 = Outcomes {c['n_outcomes_nodes']}개, "
          f"B1 = 형제 삼중쌍 {c['n_triples']}개. 중복 발화: {c['duplicate_narratives']['n_duplicate_groups']}그룹 / "
          f"{c['duplicate_narratives']['total']}건 중 고유 {c['duplicate_narratives']['unique']}건.")
        a("")
    a(f"- **A2 (근거 보유율)**: {fmt(c['A2']['score'])}")
    a(f"- **M1 (수렴도)**: {fmt(c['M1']['score'])} (trend={fmt(c['M1']['trend'])})")
    a(f"- **M2 (값 인용률)**: {fmt(c['M2']['score'])}")
    a(f"- **D1 (역할 설명 분산, ICC_role)**: {fmt(c['D1']['ICC_role'])} (ICC_run={fmt(c['D1']['ICC_run'])})")
    a(f"- **D2 (의미 분산)**: {fmt(c['D2']['score'])}")
    a(f"- **DM1 (상호작용 구조 다양성)**: {fmt(c['DM1']['score'])}, topology={c['DM1']['topology']}")
    a(f"- **게이트**: {_gates_line(c['gates'])}")
    a("")
    a("  **X1 (예측 오차, GT 대비 APE)**")
    a("")
    for line in _x1_table(c["X1"]):
        a("  " + line)
    a("")
    a("  **X4 (Calibration)**")
    a("")
    for line in _x4_lines(c["X4"]):
        a(line)
    a("")
    a("  **DX1 (결과 공간 폭, CV)**")
    a("")
    for var, dx in c["DX1"].items():
        if dx is None:
            a(f"  - `{var}`: N/A")
        else:
            a(f"  - `{var}`: CV={fmt(dx['CV'])}, range=[{fmt(dx['min'],1)}, {fmt(dx['max'],1)}], "
              f"IQR={fmt(dx['iqr'],1)}, mean={fmt(dx['mean'],1)}")
    a("")
    if is_branch:
        a("  **B1 (분기 라벨 충실도)**")
        a("")
        for var, b1 in c["B1"].items():
            if b1.get("status") != "OK":
                a(f"  - `{var}`: {b1.get('status')}")
                continue
            warn = f" ⚠️ {b1['direction_warning']}" if b1.get("direction_warning") else ""
            a(f"  - `{var}`: B1={fmt(b1['B1'])} (direction={b1['direction']}, 위반 {b1['n_violations']}/{b1['n_triples']}, "
              f"Friedman χ²={fmt(b1['friedman_chi2'],2)}, p={fmt(b1['p'],4)}), "
              f"medians: enabling={fmt(b1['medians']['enabling'],2)} / baseline={fmt(b1['medians']['baseline'],2)} / "
              f"constraining={fmt(b1['medians']['constraining'],2)}{warn}")
        a("")
        a("  **n_eff (상관 보정, Outcomes-27 표본)**")
        a("")
        for var, ne in c["n_eff"].items():
            a(f"  - `{var}`: n=27 -> n_eff={fmt(ne['n_eff'],1)} (ICC_subtree={fmt(ne['ICC_subtree'])})")
        a("")


def build():
    d = json.load(open(ROOT / "outputs" / "evaluation.json", encoding="utf-8"))
    lines = []
    a = lines.append

    a("# 정책 시뮬레이션 발화 평가 결과 (v4: 100E / BK21 / NSA / K-Pass)")
    a("")
    a(f"생성 시각: {d['meta']['generated_at']} | seed: {d['meta']['seed']} | "
      f"retriever: {d['meta']['retriever_backend']} (tau={d['meta']['embedding_threshold']})")
    a("")
    a("> **A1(문서 모순)은 이 리포트에서 제외합니다.** v4 §6의 수기 검토 프로토콜(모순 판정 100건을 "
      "사람이 A/B/C로 라벨링)을 신뢰성 있게 수행하려면, 지금처럼 20~40개 사실을 수기로 고른 fact_table "
      "기반 별칭 매칭 방식이 아니라 문서 내 모든 수치와 발화 내 모든 언급 수치를 전수 추출해 대조하는 "
      "방식이 필요하다는 판단하에 A1 자체를 결과에서 제외하기로 함(사용자 결정). "
      "`evaluation.json`에는 원시 A1 계산값이 남아있지만 신뢰할 수 있는 결과로 취급하지 않습니다. "
      "A2(근거 보유율)는 A1과 별개의 계산(문서/KG/이전발화 대비 근거 출처 분류)이라 그대로 포함합니다.")
    a("")

    a("## 주요 발견 요약")
    a("")
    a("1. **BK21 공식 타깃 변수는 사실 퍼센트가 아니라 명(headcount)이었음**: 정부가 지정한 시뮬레이션 타깃 변수명 "
      "(`graduate_employment_rate_of_bk21_participants_in_2022` 등)의 값이 [0,100] 범위를 83%가 벗어나 처음엔 "
      "'단위 오독'으로 추정했으나, 사용자가 이상치 시나리오를 직접 확인한 결과 실제로는 취업자/협력 건수(사람 수·건수)를 "
      "나타내고 있음을 확인. 실제 퍼센트 값은 같은 phase의 동일 개념 변수(`employment_rate`: 72~90%, "
      "`international_collaboration_rate`: 10~85%)에 담겨 있었고, GT(82.2%/36.0%)와도 각각 APE 4.0%/32.1%로 근접. "
      "v4에서는 `headcount_mislabeled`를 공식 변수 타입으로 신설해 X1/X4/D1/D2/DX1/G2 전 지표에서 자동 제외되도록 함.")
    a("2. **v3의 M1 집계 불일치 버그 수정**: v3 리포트는 셀 요약(§4)과 셀 간 대비(§5)에서 서로 다른 집계 방식을 써 "
      "같은 셀의 M1이 두 값(0.978 vs 0.844)으로 갈렸음. run별로 계산 후 평균하는 방식으로 통일해 이제 두 표에서 "
      "항상 같은 값이 나옴을 직접 검증(재현 테스트 포함).")
    a("3. **다국어 임베딩 재보정**: BK21/NSA/K-Pass 등 언어 불일치 가능성이 있는 정책을 위해 TF-IDF 대신 다국어 "
      "sentence-embedding을 KG tier·A2 근거 프로파일에 도입. 최초 시도에서 TF-IDF용 임계값(0.35)을 그대로 쓰자 "
      "GroundedTotal이 전 셀 0.98~0.99로 치솟는 부작용 발생 -- 정책 간 교차 null 분포(중앙값 0.43, p95 0.557)로 "
      "재보정(0.55)하고 prior 풀도 동일 백엔드로 통일한 뒤에야 0.7~0.8대의 합리적 범위로 복귀.")
    a("4. **BK21 KG는 T1=0**: source_spans/edge 텍스트가 전부 영어인데 원문 문서는 한국어라 문자열 일치가 원천적으로 "
      "불가능. 다국어 임베딩으로 개선되는 것은 T2(의미적 정렬)뿐이고 T1은 번역해도 해결 안 됨.")
    a("5. **정책 효과 대비(100E vs BK21)에는 BK21의 KG 열위라는 교란요인이 있음**: BK21 KG는 100E의 1/4 규모(49 vs "
      "218 노드)이고 T1이 0개라, 두 정책 간 차이를 '정책 유형 효과'로 해석하면 안 됨을 contrasts에 "
      "confound_note로 명시.")
    a("6. **분기(NSA/K-Pass) 데이터로 v4 신설 지표 검증**: 121노드 트리 구조, 중복 발화(NSA 605건 중 고유 582건)가 "
      "가이드라인이 실측했다고 밝힌 수치와 정확히 일치 -- 로더 구현이 독립적으로 같은 값을 재현.")
    a("7. **NSA 에이전트는 정책 효과를 체계적으로 과소예측**: 사용자가 확인해준 실제 GT(응급 15%p / 비응급 11%p 감소) "
      "대비 X4 PIT=1.000(양쪽 다) -- Outcomes 27개 중 100%가 실제값 이하로, 무작위 오차가 아니라 일관된 과소예측 "
      "패턴. APE도 40.3%/50.7%로 K-Pass(10.9~54.6%, 대체로 더 근접)보다 전반적으로 큼.")
    a("")

    a("## 0. 셀 구성")
    a("")
    a(d["meta"]["cell_design_correction"])
    a(f"\n분기 셀: {', '.join(d['meta'].get('branch_cells', []))}")
    a("")

    a("## 1. 데이터 적재 검증")
    a("")
    a("| 셀 | n_runs | revise_noop 검증 | aggregated_value==평균 검증 |")
    a("|---|---|---|---|")
    for cid, lr in d["meta"]["load_reports"].items():
        a(f"| {cid} | {d['by_cell'][cid]['n_runs']} | {lr['revise_noop_verified']} | {lr['aggregated_value_matches_mean']} |")
    a("")

    a("## 2. KG 품질 (tier 분포, 다국어 임베딩 재보정 적용 후)")
    a("")
    a("| 정책 | nodes | edges | literal_match_rate | align_median | align_tfidf | align_embedding | T1 | T2 | T3 |")
    a("|---|---|---|---|---|---|---|---|---|---|")
    for pid, k in d["meta"]["kg_meta"].items():
        tc = k["tier_counts"]
        a(f"| {pid} | {k['n_nodes']} | {k['n_edges']} | {k['kg_span_literal_match_rate']:.2%} | "
          f"{k['chunk_alignment_confidence_median']:.3f} | {k.get('chunk_alignment_confidence_median_tfidf', float('nan')):.3f} | "
          f"{k.get('chunk_alignment_confidence_median_embedding', float('nan')):.3f} | {tc['T1']} | {tc['T2']} | {tc['T3']} |")
    a("")
    a("(NSA/K-Pass에는 별도 KG가 없어 이 표에는 나타나지 않습니다.)")
    a("")

    a("## 3. 변수 유형 분류")
    a("")
    a("| 정책 | anchored_constant | predicted | unanchored_constant | headcount_mislabeled |")
    a("|---|---|---|---|---|")
    for pid, vt in d["meta"]["variable_typing"].items():
        a(f"| {pid} | {vt['anchored']} | {vt['predicted']} | {vt['unanchored']} | {len(vt.get('headcount_mislabeled', []))} |")
    for pid, vt in d["meta"]["variable_typing"].items():
        if vt.get("headcount_mislabeled"):
            a(f"\n`{pid}` headcount_mislabeled: {', '.join(vt['headcount_mislabeled'])}")
    a("")

    a("## 4. 셀별 핵심 지표 (단일 pathway)")
    a("")
    for cid, c in d["by_cell"].items():
        _cell_section(a, cid, c, is_branch=False)

    a("## 5. 셀 간 대비 (Contrasts)")
    a("")
    for name, c in d["contrasts"].items():
        if c is None:
            a(f"- **{name}**: 추정 불가 (데이터 없음)")
            continue
        a(f"### {name}: {c['cell_a']} vs {c['cell_b']}")
        a("")
        if c.get("confound_note"):
            a(f"> ⚠️ {c['confound_note']}")
            a("")
        for metric in ["M1", "D2"]:
            m = c[metric]
            sig = "✅ 유의미" if m["delta_CI95"] and not (m["delta_CI95"][0] <= 0 <= m["delta_CI95"][1]) else "비유의"
            a(f"- **{metric}**: {c['cell_a']}={fmt(m['a_mean'])}, {c['cell_b']}={fmt(m['b_mean'])}, "
              f"Δ={fmt(m['delta'])} 95%CI=[{fmt(m['delta_CI95'][0]) if m['delta_CI95'] else 'N/A'}, "
              f"{fmt(m['delta_CI95'][1]) if m['delta_CI95'] else 'N/A'}] -- {sig}")
        if "X1_APE" in c:
            for var, m in c["X1_APE"].items():
                sig = "✅ 유의미" if m["delta_CI95"] and not (m["delta_CI95"][0] <= 0 <= m["delta_CI95"][1]) else "비유의"
                a(f"- **X1_APE[{var}]**: Δ={fmt(m['delta'],1)}%p, 95%CI=[{fmt(m['delta_CI95'][0],1)}, "
                  f"{fmt(m['delta_CI95'][1],1)}] -- {sig}")
        a("")

    if d.get("by_branch_cell"):
        a("## 6. 분기(assumption-conditioned) pathway -- NSA / K-Pass")
        a("")
        a("보조 대상. §7 실험 설계와 무관한 예비 검증용 (경로 구조·모델이 통제된 nsa_deepseek vs kpass_deepseek 비교).")
        a("")
        for cid, c in d["by_branch_cell"].items():
            _cell_section(a, cid, c, is_branch=True)

    if d.get("branch_contrasts"):
        a("## 7. 분기 셀 간 대비 (정책 유형)")
        a("")
        for group_name, group in d["branch_contrasts"].items():
            for name, c in group.items():
                a(f"### {name}: {c['cell_a']} vs {c['cell_b']}")
                a("")
                if c.get("note"):
                    a(f"> {c['note']}")
                    a("")
                m1 = c["M1"]
                sig = "✅ 유의미" if m1["delta_CI95"] and not (m1["delta_CI95"][0] <= 0 <= m1["delta_CI95"][1]) else "비유의"
                a(f"- **M1**: {c['cell_a']}={fmt(m1['a_mean'])}, {c['cell_b']}={fmt(m1['b_mean'])}, "
                  f"Δ={fmt(m1['delta'])} 95%CI=[{fmt(m1['delta_CI95'][0]) if m1['delta_CI95'] else 'N/A'}, "
                  f"{fmt(m1['delta_CI95'][1]) if m1['delta_CI95'] else 'N/A'}] -- {sig}")
                a(f"- **D1 (ICC_role, 점비교)**: {c['cell_a']}={fmt(c['D1_ICC_role']['a'])}, "
                  f"{c['cell_b']}={fmt(c['D1_ICC_role']['b'])}")
                a(f"- **DM1 (점비교)**: {c['cell_a']}={fmt(c['DM1']['a'])}, {c['cell_b']}={fmt(c['DM1']['b'])}")
                a("")

    out_path = ROOT / "outputs" / "report.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    build()
