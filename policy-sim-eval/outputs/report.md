# 정책 시뮬레이션 발화 평가 결과 (v4: 100E / BK21 / NSA / K-Pass)

생성 시각: 2026-08-10T23:18:26 | seed: 20260810 | retriever: embedding (tau=0.55)

> **A1(문서 모순)은 이 리포트에서 제외합니다.** v4 §6의 수기 검토 프로토콜(모순 판정 100건을 사람이 A/B/C로 라벨링)을 신뢰성 있게 수행하려면, 지금처럼 20~40개 사실을 수기로 고른 fact_table 기반 별칭 매칭 방식이 아니라 문서 내 모든 수치와 발화 내 모든 언급 수치를 전수 추출해 대조하는 방식이 필요하다는 판단하에 A1 자체를 결과에서 제외하기로 함(사용자 결정). `evaluation.json`에는 원시 A1 계산값이 남아있지만 신뢰할 수 있는 결과로 취급하지 않습니다. A2(근거 보유율)는 A1과 별개의 계산(문서/KG/이전발화 대비 근거 출처 분류)이라 그대로 포함합니다.

## 주요 발견 요약

1. **BK21 공식 타깃 변수는 사실 퍼센트가 아니라 명(headcount)이었음**: 정부가 지정한 시뮬레이션 타깃 변수명 (`graduate_employment_rate_of_bk21_participants_in_2022` 등)의 값이 [0,100] 범위를 83%가 벗어나 처음엔 '단위 오독'으로 추정했으나, 사용자가 이상치 시나리오를 직접 확인한 결과 실제로는 취업자/협력 건수(사람 수·건수)를 나타내고 있음을 확인. 실제 퍼센트 값은 같은 phase의 동일 개념 변수(`employment_rate`: 72~90%, `international_collaboration_rate`: 10~85%)에 담겨 있었고, GT(82.2%/36.0%)와도 각각 APE 4.0%/32.1%로 근접. v4에서는 `headcount_mislabeled`를 공식 변수 타입으로 신설해 X1/X4/D1/D2/DX1/G2 전 지표에서 자동 제외되도록 함.
2. **v3의 M1 집계 불일치 버그 수정**: v3 리포트는 셀 요약(§4)과 셀 간 대비(§5)에서 서로 다른 집계 방식을 써 같은 셀의 M1이 두 값(0.978 vs 0.844)으로 갈렸음. run별로 계산 후 평균하는 방식으로 통일해 이제 두 표에서 항상 같은 값이 나옴을 직접 검증(재현 테스트 포함).
3. **다국어 임베딩 재보정**: BK21/NSA/K-Pass 등 언어 불일치 가능성이 있는 정책을 위해 TF-IDF 대신 다국어 sentence-embedding을 KG tier·A2 근거 프로파일에 도입. 최초 시도에서 TF-IDF용 임계값(0.35)을 그대로 쓰자 GroundedTotal이 전 셀 0.98~0.99로 치솟는 부작용 발생 -- 정책 간 교차 null 분포(중앙값 0.43, p95 0.557)로 재보정(0.55)하고 prior 풀도 동일 백엔드로 통일한 뒤에야 0.7~0.8대의 합리적 범위로 복귀.
4. **BK21 KG는 T1=0**: source_spans/edge 텍스트가 전부 영어인데 원문 문서는 한국어라 문자열 일치가 원천적으로 불가능. 다국어 임베딩으로 개선되는 것은 T2(의미적 정렬)뿐이고 T1은 번역해도 해결 안 됨.
5. **정책 효과 대비(100E vs BK21)에는 BK21의 KG 열위라는 교란요인이 있음**: BK21 KG는 100E의 1/4 규모(49 vs 218 노드)이고 T1이 0개라, 두 정책 간 차이를 '정책 유형 효과'로 해석하면 안 됨을 contrasts에 confound_note로 명시.
6. **분기(NSA/K-Pass) 데이터로 v4 신설 지표 검증**: 121노드 트리 구조, 중복 발화(NSA 605건 중 고유 582건)가 가이드라인이 실측했다고 밝힌 수치와 정확히 일치 -- 로더 구현이 독립적으로 같은 값을 재현.
7. **NSA 에이전트는 정책 효과를 체계적으로 과소예측**: 사용자가 확인해준 실제 GT(응급 15%p / 비응급 11%p 감소) 대비 X4 PIT=1.000(양쪽 다) -- Outcomes 27개 중 100%가 실제값 이하로, 무작위 오차가 아니라 일관된 과소예측 패턴. APE도 40.3%/50.7%로 K-Pass(10.9~54.6%, 대체로 더 근접)보다 전반적으로 큼.

## 0. 셀 구성

guideline's reference cell table names the third cell 'bk21_gemini'; actual data on disk is DeepSeek-generated (run_log.json). Cells here are 100e_gemini/100e_deepseek/bk21_deepseek -- see configs/cells.yaml header.

분기 셀: nsa_deepseek, kpass_deepseek

## 1. 데이터 적재 검증

| 셀 | n_runs | revise_noop 검증 | aggregated_value==평균 검증 |
|---|---|---|---|
| 100e_gemini | 50 | True | True |
| 100e_deepseek | 50 | True | True |
| bk21_deepseek | 50 | True | True |

## 2. KG 품질 (tier 분포, 다국어 임베딩 재보정 적용 후)

| 정책 | nodes | edges | literal_match_rate | align_median | align_tfidf | align_embedding | T1 | T2 | T3 |
|---|---|---|---|---|---|---|---|---|---|
| 100e | 218 | 280 | 56.16% | 0.815 | 0.686 | 0.807 | 293 | 197 | 8 |
| bk21 | 49 | 135 | 0.00% | 0.675 | 0.232 | 0.667 | 0 | 171 | 13 |

(NSA/K-Pass에는 별도 KG가 없어 이 표에는 나타나지 않습니다.)

## 3. 변수 유형 분류

| 정책 | anchored_constant | predicted | unanchored_constant | headcount_mislabeled |
|---|---|---|---|---|
| 100e | 2 | 17 | 2 | 0 |
| bk21 | 0 | 16 | 0 | 2 |
| nsa | 0 | 15 | 0 | 0 |
| kpass | 0 | 16 | 0 | 0 |

`bk21` headcount_mislabeled: Outcomes.international_research_collaboration_rate_of_bk21_participants_in_2022, Outcomes.graduate_employment_rate_of_bk21_participants_in_2022

## 4. 셀별 핵심 지표 (단일 pathway)

### 100e_gemini

- **A2 (근거 보유율)**: 0.828
- **M1 (수렴도)**: 0.945 (trend=0.482)
- **M2 (값 인용률)**: 0.404
- **D1 (역할 설명 분산, ICC_role)**: 0.035 (ICC_run=0.961)
- **D2 (의미 분산)**: 0.335
- **DM1 (상호작용 구조 다양성)**: 0.751, topology={'disconnected': 56, 'star': 7, 'chain': 40, 'complete': 147}
- **게이트**: 체인 일관성 실패(위반 6/200) · 값 범위 통과(위반 0/1250) · 완료율 100.0%

  **X1 (예측 오차, GT 대비 APE)**

  | 타깃 | true_value | source | APE mean | APE median | 95% CI |
  |---|---|---|---|---|---|
  | number_of_100e_projects_approved_in_2023 | 115 | direct | 30.9% | 15.2% | [22.6, 40.4]% |
  | deployed_100e_projects | 73 | direct | 39.9% | 31.2% | [31.1, 51.4]% |
  | approved_100e_projects | 115 | concept_duplicate | 30.9% | 15.2% | [22.2, 40.5]% |

  **X4 (Calibration)**

  - `number_of_100e_projects_approved_in_2023`: coverage_50=True, coverage_90=True, PIT=0.500, CRPS=9.1
  - `deployed_100e_projects`: coverage_50=True, coverage_90=True, PIT=0.440, CRPS=8.4

  **DX1 (결과 공간 폭, CV)**

  - `number_of_100e_projects_approved_in_2023`: CV=0.451, range=[2.0, 300.0], IQR=32.8, mean=113.9
  - `deployed_100e_projects`: CV=0.500, range=[0.0, 219.8], IQR=44.3, mean=77.3

### 100e_deepseek

- **A2 (근거 보유율)**: 0.785
- **M1 (수렴도)**: 0.939 (trend=0.668)
- **M2 (값 인용률)**: 0.354
- **D1 (역할 설명 분산, ICC_role)**: 0.028 (ICC_run=0.958)
- **D2 (의미 분산)**: 0.254
- **DM1 (상호작용 구조 다양성)**: 0.649, topology={'disconnected': 80, 'star': 2, 'chain': 18, 'complete': 150}
- **게이트**: 체인 일관성 통과(위반 0/200) · 값 범위 통과(위반 0/1250) · 완료율 100.0%

  **X1 (예측 오차, GT 대비 APE)**

  | 타깃 | true_value | source | APE mean | APE median | 95% CI |
  |---|---|---|---|---|---|
  | number_of_100e_projects_approved_in_2023 | 115 | direct | 32.4% | 30.4% | [27.2, 38.3]% |
  | deployed_100e_projects | 73 | direct | 41.7% | 37.3% | [35.7, 47.6]% |
  | approved_100e_projects | 115 | concept_duplicate | 32.4% | 30.4% | [27.3, 38.1]% |

  **X4 (Calibration)**

  - `number_of_100e_projects_approved_in_2023`: coverage_50=False, coverage_90=False, PIT=1.000, CRPS=24.5
  - `deployed_100e_projects`: coverage_50=False, coverage_90=False, PIT=1.000, CRPS=22.0

  **DX1 (결과 공간 폭, CV)**

  - `number_of_100e_projects_approved_in_2023`: CV=0.303, range=[25.0, 104.0], IQR=40.0, mean=77.7
  - `deployed_100e_projects`: CV=0.355, range=[7.6, 71.0], IQR=19.8, mean=42.5

### bk21_deepseek

- **A2 (근거 보유율)**: 0.730
- **M1 (수렴도)**: 0.845 (trend=0.077)
- **M2 (값 인용률)**: 0.350
- **D1 (역할 설명 분산, ICC_role)**: 0.047 (ICC_run=0.935)
- **D2 (의미 분산)**: 0.185
- **DM1 (상호작용 구조 다양성)**: 0.587, topology={'disconnected': 110, 'star': 2, 'chain': 5, 'complete': 133}
- **게이트**: 체인 일관성 실패(위반 6/150) · 값 범위 통과(위반 0/1250) · 완료율 100.0%

  **X1 (예측 오차, GT 대비 APE)**

  | 타깃 | true_value | source | APE mean | APE median | 95% CI |
  |---|---|---|---|---|---|
  | employment_rate | 82.2 | direct | 4.0% | 3.4% | [3.2, 4.8]% |
  | international_collaboration_rate | 36.0 | direct | 32.1% | 27.8% | [26.1, 38.6]% |

  **X4 (Calibration)**

  - `employment_rate`: coverage_50=True, coverage_90=True, PIT=0.340, CRPS=1.1
  - `international_collaboration_rate`: coverage_50=True, coverage_90=True, PIT=0.600, CRPS=3.7

  **DX1 (결과 공간 폭, CV)**

  - `employment_rate`: CV=0.048, range=[72.5, 90.0], IQR=5.0, mean=82.9
  - `international_collaboration_rate`: CV=0.412, range=[10.0, 70.0], IQR=15.8, mean=34.4

## 5. 셀 간 대비 (Contrasts)

### model_effect_100e: 100e_gemini vs 100e_deepseek

- **M1**: 100e_gemini=0.945, 100e_deepseek=0.939, Δ=0.006 95%CI=[-0.016, 0.026] -- 비유의
- **D2**: 100e_gemini=0.335, 100e_deepseek=0.254, Δ=0.081 95%CI=[0.062, 0.102] -- ✅ 유의미
- **X1_APE[number_of_100e_projects_approved_in_2023]**: Δ=-1.5%p, 95%CI=[-12.5, 10.7] -- 비유의
- **X1_APE[deployed_100e_projects]**: Δ=-1.8%p, 95%CI=[-13.4, 10.1] -- 비유의

### policy_effect_deepseek: 100e_deepseek vs bk21_deepseek

> ⚠️ BK21's KG has 1/4 the nodes of 100E's (49 vs 218) and 0 T1-tier facts (vs 293 for 100E) -- a KG-quality confound, not necessarily a policy-type effect. Do not interpret an A1/A2 gap here as "distributive Korean policy vs distributive Singaporean policy" without controlling for this.

- **M1**: 100e_deepseek=0.939, bk21_deepseek=0.845, Δ=0.093 95%CI=[0.049, 0.144] -- ✅ 유의미
- **D2**: 100e_deepseek=0.254, bk21_deepseek=0.185, Δ=0.069 95%CI=[0.046, 0.090] -- ✅ 유의미

- **interaction**: 추정 불가 (데이터 없음)
## 6. 분기(assumption-conditioned) pathway -- NSA / K-Pass

보조 대상. §7 실험 설계와 무관한 예비 검증용 (경로 구조·모델이 통제된 nsa_deepseek vs kpass_deepseek 비교).

### nsa_deepseek

표본 단위: A2/M1/M2/D1/D2/DM1 = leaf 81개, X1/X4/DX1 = Outcomes 27개, B1 = 형제 삼중쌍 40개. 중복 발화: 18그룹 / 605건 중 고유 582건.

- **A2 (근거 보유율)**: 0.726
- **M1 (수렴도)**: 0.690 (trend=0.031)
- **M2 (값 인용률)**: 0.140
- **D1 (역할 설명 분산, ICC_role)**: 0.646 (ICC_run=0.354)
- **D2 (의미 분산)**: 0.452
- **DM1 (상호작용 구조 다양성)**: 0.600, topology={'disconnected': 235, 'star': 0, 'chain': 20, 'complete': 150}
- **게이트**: 체인 일관성 해당없음 · 값 범위 통과(위반 0/2835) · 완료율 100.0%

  **X1 (예측 오차, GT 대비 APE)**

  | 타깃 | true_value | source | APE mean | APE median | 95% CI |
  |---|---|---|---|---|---|
  | emergency_oon_bill_prevalence_reduction_pp | 15 | direct | 40.3% | 43.3% | [33.2, 46.6]% |
  | nonemergency_oon_bill_prevalence_reduction_pp | 11 | direct | 50.7% | 52.4% | [44.0, 57.4]% |

  **X4 (Calibration)**

  - `emergency_oon_bill_prevalence_reduction_pp`: coverage_50=False, coverage_90=False, PIT=1.000, CRPS=4.6
  - `nonemergency_oon_bill_prevalence_reduction_pp`: coverage_50=False, coverage_90=False, PIT=1.000, CRPS=4.4

  **DX1 (결과 공간 폭, CV)**

  - `emergency_oon_bill_prevalence_reduction_pp`: CV=0.295, range=[4.0, 15.0], IQR=3.1, mean=8.9
  - `nonemergency_oon_bill_prevalence_reduction_pp`: CV=0.376, range=[2.0, 10.2], IQR=2.1, mean=5.4

  **B1 (분기 라벨 충실도)**

  - `emergency_oon_bill_prevalence_reduction_pp`: B1=0.667 (direction=higher_is_better, 위반 3/9, Friedman χ²=12.67, p=0.0018), medians: enabling=9.30 / baseline=9.80 / constraining=7.40
  - `nonemergency_oon_bill_prevalence_reduction_pp`: B1=0.667 (direction=higher_is_better, 위반 3/9, Friedman χ²=12.67, p=0.0018), medians: enabling=5.92 / baseline=5.80 / constraining=3.40
  - `patient_savings_per_protected_claim_usd`: B1=0.889 (direction=higher_is_better, 위반 1/9, Friedman χ²=15.60, p=0.0004), medians: enabling=670.00 / baseline=516.00 / constraining=330.00

  **n_eff (상관 보정, Outcomes-27 표본)**

  - `emergency_oon_bill_prevalence_reduction_pp`: n=27 -> n_eff=10.2 (ICC_subtree=0.205)
  - `nonemergency_oon_bill_prevalence_reduction_pp`: n=27 -> n_eff=16.1 (ICC_subtree=0.084)
  - `patient_savings_per_protected_claim_usd`: n=27 -> n_eff=18.1 (ICC_subtree=0.062)

### kpass_deepseek

표본 단위: A2/M1/M2/D1/D2/DM1 = leaf 81개, X1/X4/DX1 = Outcomes 27개, B1 = 형제 삼중쌍 40개. 중복 발화: 22그룹 / 605건 중 고유 559건.

- **A2 (근거 보유율)**: 0.649
- **M1 (수렴도)**: 0.476 (trend=0.266)
- **M2 (값 인용률)**: 0.068
- **D1 (역할 설명 분산, ICC_role)**: 0.717 (ICC_run=0.283)
- **D2 (의미 분산)**: 0.784
- **DM1 (상호작용 구조 다양성)**: 0.476, topology={'disconnected': 319, 'star': 0, 'chain': 35, 'complete': 51}
- **게이트**: 체인 일관성 통과(위반 0/81) · 값 범위 통과(위반 0/405) · 완료율 100.0%

  **X1 (예측 오차, GT 대비 APE)**

  | 타깃 | true_value | source | APE mean | APE median | 95% CI |
  |---|---|---|---|---|---|
  | cumulative_registered_kpass_users_count | 3000000 | direct | 54.6% | 53.5% | [50.3, 59.5]% |
  | avg_monthly_refund_per_user_krw | 18000 | direct | 10.9% | 10.9% | [8.0, 13.7]% |
  | avg_refund_share_of_transit_spending_pct | 26.6 | direct | 13.3% | 12.4% | [11.7, 15.0]% |
  | low_income_avg_monthly_refund_per_user_krw | 37000 | direct | 13.3% | 11.3% | [10.1, 16.2]% |

  **X4 (Calibration)**

  - `cumulative_registered_kpass_users_count`: coverage_50=False, coverage_90=False, PIT=1.000, CRPS=1428650.2
  - `avg_monthly_refund_per_user_krw`: coverage_50=True, coverage_90=True, PIT=0.630, CRPS=707.9
  - `avg_refund_share_of_transit_spending_pct`: coverage_50=False, coverage_90=False, PIT=1.000, CRPS=2.9
  - `low_income_avg_monthly_refund_per_user_krw`: coverage_50=True, coverage_90=True, PIT=0.667, CRPS=2019.7

  **DX1 (결과 공간 폭, CV)**

  - `cumulative_registered_kpass_users_count`: CV=0.266, range=[802000.0, 2040000.0], IQR=513000.0, mean=1363481.5
  - `avg_monthly_refund_per_user_krw`: CV=0.129, range=[12130.0, 20600.0], IQR=3080.0, mean=17023.4
  - `avg_refund_share_of_transit_spending_pct`: CV=0.051, range=[20.0, 25.3], IQR=1.4, mean=23.1
  - `low_income_avg_monthly_refund_per_user_krw`: CV=0.150, range=[25940.0, 46800.0], IQR=7620.0, mean=34145.2

  **B1 (분기 라벨 충실도)**

  - `avg_monthly_refund_per_user_krw`: B1=0.444 (direction=higher_is_better, 위반 5/9, Friedman χ²=6.89, p=0.0319), medians: enabling=18600.00 / baseline=16660.00 / constraining=16240.00
  - `avg_refund_share_of_transit_spending_pct`: B1=0.778 (direction=higher_is_better, 위반 2/9, Friedman χ²=10.89, p=0.0043), medians: enabling=23.80 / baseline=23.00 / constraining=22.20
  - `cumulative_registered_kpass_users_count`: B1=1.000 (direction=higher_is_better, 위반 0/9, Friedman χ²=18.00, p=0.0001), medians: enabling=1484000 / baseline=1394000 / constraining=1194000
  - `low_income_avg_monthly_refund_per_user_krw`: B1=0.778 (direction=higher_is_better, 위반 2/9, Friedman χ²=14.89, p=0.0006), medians: enabling=37800.00 / baseline=34500.00 / constraining=29900.00

  **n_eff (상관 보정, Outcomes-27 표본)**

  - `avg_monthly_refund_per_user_krw`: n=27 -> n_eff=7.6 (ICC_subtree=0.316)
  - `avg_refund_share_of_transit_spending_pct`: n=27 -> n_eff=24.5 (ICC_subtree=0.013)
  - `cumulative_registered_kpass_users_count`: n=27 -> n_eff=3.6 (ICC_subtree=0.811)
  - `low_income_avg_monthly_refund_per_user_krw`: n=27 -> n_eff=8.5 (ICC_subtree=0.274)

## 7. 분기 셀 간 대비 (정책 유형)

### nsa_vs_kpass: nsa_deepseek vs kpass_deepseek

> Path structure and model are controlled between these two cells (both single DeepSeek-generated 121-node trees), so a difference here is attributable to policy TYPE (regulatory vs distributive), not design or model. Preliminary interpretive-framework validation only (n=1 tree per policy) -- primary comparison metrics: M1, D1, DM1.

- **M1**: nsa_deepseek=0.690, kpass_deepseek=0.476, Δ=0.214 95%CI=[0.186, 0.241] -- ✅ 유의미
- **D1 (ICC_role, 점비교)**: nsa_deepseek=0.646, kpass_deepseek=0.717
- **DM1 (점비교)**: nsa_deepseek=0.600, kpass_deepseek=0.476
