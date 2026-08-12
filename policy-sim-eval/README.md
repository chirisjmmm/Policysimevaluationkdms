# policy-sim-eval

`EVAL_GUIDELINE.md` (v3) 기반 정책 시뮬레이션 발화 평가 파이프라인 구현.
100E(싱가포르)·BK21(한국) 3개 셀(100e_gemini, 100e_deepseek, bk21_deepseek), 총 150개 시나리오에 대해
Accuracy(A1/A2, M1-M3, X1-X5) / Diversity(D1/D2, DM1/DM2, DX1-DX3) 지표를 로컬 결정적 연산으로 계산합니다.
LLM-as-judge, 외부 유료 API 없음.

## 실행

```bash
pip install numpy scipy pandas scikit-learn networkx rapidfuzz pyyaml pytest sentence-transformers
python -m pytest tests/            # normalize.py 경계값 테스트 11건
python -m src.cli                  # 전체 파이프라인 실행 -> outputs/evaluation.json (A2 임베딩 계산 때문에 ~30분 소요)
python -m src.report.build_report  # outputs/report.md 요약 리포트 생성
```

기본 retriever 백엔드는 `embedding`(다국어 sentence-transformers, `configs/eval_default.yaml`에서 설정)입니다.
최초 실행 시 `paraphrase-multilingual-MiniLM-L12-v2` 모델(~470MB)을 HuggingFace Hub에서 내려받습니다(로컬 캐시 후 재사용, 무료).
인터넷 연결이 없거나 모델 로드가 실패하면 TF-IDF로 자동 폴백됩니다.

## 구조

```
configs/          cells.yaml, policy_100e.yaml, policy_bk21.yaml, eval_default.yaml
data/policies/    {100e,bk21}/{doc.md, graph.json, runs/{model}/scenario_*.json}
src/
  io/             normalize.py(경계인지 수치파싱) ir.py loaders.py
  prep/           doc_clean.py(오프셋맵) kg_clean.py var_typing.py
  index/          doc_index.py kg_index.py(tier 판정) retriever.py(TF-IDF+다국어 임베딩, 항상 둘 다 계산해 보고)
  metrics/        acc_micro.py(A1,A2) acc_meso.py(M1-M3) acc_macro.py(X1-X5)
                  div_micro.py(D1,D2) div_meso.py(DM1,DM2) div_macro.py(DX1-DX3)
  design/         cells.py contrasts.py(부트스트랩 대비)
  report/         build_report.py
tests/            test_normalize.py
outputs/          evaluation.json report.md
```

## 가이드라인과 다른 점 (실측 기반 수정)

1. **셀 구성**: 가이드라인은 `bk21_gemini`를 3번째 셀로 명시하지만, 실제 데이터는 DeepSeek로 생성됨
   (`run_log.json`: `"model": "deepseek-chat"`). 셀은 `100e_gemini` / `100e_deepseek` / `bk21_deepseek`.
   정책 효과 대비는 DeepSeek 기준으로만 추정 가능(Gemini 기준 아님).
2. **BK21 KG는 원문과 언어 불일치**: KG의 summary/source_spans/edge text가 전부 영어, 원문 문서는 한국어.
   → T1(문자열 일치) 근거 0건. tier 판정은 evidence_alignment 속성(모든 edge가 "supported"로 균일해 무의미)
   대신 원문 대조(리터럴/유사도)로 직접 재계산.
3. **BK21 시나리오 발화도 영어**: fact_table 별칭에 영어 표현을 병기하지 않으면 A1이 0건 매칭.
4. **`Inputs.total_funding_allocated`(BK21)는 전국 총예산이 아니라 시나리오별 특정 대학 배분액** →
   anchored_constant 아님, predicted로 재분류.
5. **X1 실제값(GT)**: 사용자가 제공한 Policy A/B 비교표 사용 (100E: 승인 115 / 시행 73 [2023 누적],
   BK21: 취업률 82.2% / 국제협력률 36.0% [2022]).
6. **KG tier·A2 근거 프로파일에 다국어 임베딩 도입 + 임계값 재보정**: TF-IDF는 언어가 다르면 매칭 자체가
   불가능해 KG tier(#2)와 동일한 문제를 A2도 그대로 안고 있었음. 다국어 sentence-transformers로 교체했으나,
   TF-IDF용으로 고른 기존 임계값(tau_grounding=0.35, T2 임계값=0.5)을 그대로 쓰면 신경망 임베딩의 훨씬 높은
   "무관 문장 간 유사도 바닥"(정책 간 교차 검증: 무관한데도 중앙값 0.43, p95 0.557) 때문에 100E까지 포함해
   전 셀 근거율이 비정상적으로 치솟는 부작용 발생. 정책 간 교차(서로 무관 보장) null 분포의 p95(≈0.55)로
   두 임계값을 재보정하고, A2의 prior 풀 비교도 TF-IDF에서 동일 임베딩 백엔드로 통일한 뒤에야 정상 범위로
   복귀 (`src/index/retriever.py`, `src/index/kg_index.py` 상단 주석에 방법론 상세 기록).

7. **BK21 X1/X3/X4 타깃 변수 교정**: 공식 타깃 변수명(`graduate_employment_rate_of_bk21_participants_in_2022`,
   `international_research_collaboration_rate_of_bk21_participants_in_2022`)이 X5에서 범위 위반을 대량 보여
   처음엔 "단위 오독"으로 추정했으나, 사용자가 이상치 시나리오를 직접 확인해 이 변수들이 실제로는 퍼센트가 아니라
   취업자·협력 건수(headcount)임을 확인. 실제 퍼센트는 같은 phase의 `employment_rate`(72~90%),
   `international_collaboration_rate`(10~85%)에 담겨 있어, GT 비교 대상을 이 두 변수로 교체
   (APE 7,657%/8,658% → 4.0%/32.1%). headcount로 확인된 두 변수는 [0,100] 범위 검사에서 제외.

자세한 내용은 `outputs/report.md` 상단 "주요 발견 요약"과 각 config 파일의 인라인 주석 참고.
