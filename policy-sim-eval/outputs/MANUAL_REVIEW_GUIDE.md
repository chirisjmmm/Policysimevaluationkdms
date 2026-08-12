# A1 수기 검토 가이드 (v4 guideline §6)

A1(문서 모순)은 파이프라인 전체에서 **유일하게 사람 손이 필요한 지표**입니다.
이 문서 없이 A1 점수만 보고하면 안 됩니다. 아래 절차는 5개 셀
(100e_gemini, 100e_deepseek, bk21_deepseek, nsa_deepseek, kpass_deepseek)
모두에 대해 이미 Step 1까지 자동 실행되어 있고, Step 2~3만 남았습니다.

## 이미 자동으로 끝난 것 (Step 1)

`python -m src.audit.run_audit` 실행 완료. 결과물:

```
outputs/audit/
  100e_gemini_a1_diagnostic.md   100e_gemini_a1_sample.csv
  100e_deepseek_a1_diagnostic.md 100e_deepseek_a1_sample.csv
  bk21_deepseek_a1_diagnostic.md bk21_deepseek_a1_sample.csv
  nsa_deepseek_a1_diagnostic.md  nsa_deepseek_a1_sample.csv   (46건, 모순이 46건뿐이라 전량 표본)
  kpass_deepseek_a1_diagnostic.md kpass_deepseek_a1_sample.csv
```

`_a1_diagnostic.md`는 사실(fact)별 mention/with_qty/consistent/contradiction/unscored
집계표 + 자동 진단 플래그입니다. 읽기만 하면 됩니다 (Step 2를 어디부터 손대야 할지 알려줌).

## Step 2 -- 별칭 보강 (셀당 ~20분, 선택적)

진단표에서 `mention_near_zero` 플래그가 붙은 사실만 대상입니다. 해당 사실을 다루는
발화 몇 개를 열어(원본 시나리오 JSON 또는 evidence 로그) 에이전트가 실제로 쓰는 표현을
확인하고, 해당 정책의 `configs/policy_{정책}.yaml`의 `fact_table[].aliases`에 추가합니다.

**셀별로 확인이 필요한 사실 (mention=0 또는 1):**

| 셀 | 확인 필요 fact_id |
|---|---|
| bk21_deepseek | `phd_completion_scholarship_min_monthly`, `faculty_lecture_ratio_2022_target`, `total_project_duration_years`, `education_research_groups_future_talent`, `indirect_cost_cap_pct` |
| nsa_deepseek | `open_negotiation_period_days`, `idr_initiation_window_days`, `hhs_complaint_response_deadline_days`, `abd_appeal_window_days`, `erisa_extension_days` (6개 사실 중 5개가 mention=0 -- 별칭이 거의 다 안 맞고 있다는 뜻, 우선순위 최상위) |

100e_gemini/100e_deepseek/kpass_deepseek는 mention_near_zero 플래그가 없거나 적어
우선순위가 낮습니다. **NSA가 가장 시급합니다** -- 사실 6개 중 5개가 사실상 한 건도
안 잡히고 있어, 지금 상태의 NSA A1 점수는 사실상 fact 1개(`initial_payment_or_denial_deadline_days`)
만 반영합니다.

별칭 보강 후에는 `python -m src.audit.run_audit`를 다시 돌려 진단표가 개선됐는지 확인하세요.

## Step 3 -- 모순 표본 라벨링 (셀당 ~30~40분) ★ 핵심, 생략 불가

`outputs/audit/{셀}_a1_sample.csv`를 엑셀 등으로 열어 `label` 칸(마지막 열)에
**A / B / C** 중 하나를 채웁니다.

| 라벨 | 의미 | 채우는 기준 |
|---|---|---|
| **A** | 진짜 모순 | 발화가 문서 수치를 명백히 잘못 인용 (예: 문서 "최대 33만 S$" ↔ 발화 "50만 S$로 상한") |
| **B** | 스코프 파생 | 발화가 문서 값을 정확히 알고 있으면서, 그와 다른 하위 범위(연도별/기관별 등) 추정치를 별도로 제시 |
| **C** | 매칭 오류 | 별칭 근처의 무관한 숫자가 잘못 대조됨 |

각 행의 `context_text` 열(수치 앞뒤 200자)과 `doc_value`/`doc_bound`(문서 기준값),
`uttered_value`(발화에서 추출된 값)를 보고 판단합니다.

**실제 발견된 C(매칭 오류) 예시** (bk21_deepseek 표본 1번 행):
- `fact_id`: `participants_innovation_talent_annual` (문서값 6,400)
- 발화: "...With 6,400 graduate students and 205 postdocs, the total active
  participants are about 6,605. Multiplying by 2.5 gives about 16,512
  publications..."
- `uttered_value`: **6.0** -- 발화가 6,400을 정확히 인용했는데도 추출기가 근처의
  다른 소수점 값(2.5 등)을 잘못 집어 6.0으로 기록. 이런 행은 **C**로 라벨링.

라벨링이 끝나면:

```python
from src.audit.a1_sampler import compute_precision
print(compute_precision("outputs/audit/bk21_deepseek_a1_sample.csv"))
# {"n_sampled":100, "n_labeled":.., "true_contradiction_A":.., "scope_derived_B":..,
#  "match_error_C":.., "precision": |A|/n_labeled}
```

**보고 형식** (v4 §6.3): `A1 = 0.253 (모순 판정 100건 수기 검토: 진짜 모순 N건,
스코프 파생 M건, 매칭 오류 K건, precision=N/100)`. 이 한 줄 없이는 A1 점수 전체를
쓸 수 없습니다.

## Step 4 -- 배제 필터 추가 (선택, 생략 가능)

Step 3에서 B(스코프 파생) 라벨이 많이 나오면, 공통 신호어를 뽑아
`configs/policy_{정책}.yaml`에 아래와 같은 배제 목록을 추가하고 재실행하면 A1이 상승합니다.
시간이 없으면 생략하고 한계로만 명시하는 편이 낫습니다 (guideline 원문 그대로).

```yaml
scope_derivation_markers:
  - allocation | allocated for | portion of | share of
  - estimate | estimated | projected
  - "in 2023" | annual | per institution | 기관별 | 연차별
```//
(이 필터를 실제로 A1 파이프라인에 적용하는 코드는 아직 구현하지 않았습니다 -- Step 3
결과를 보고 필요하다고 판단되면 알려주시면 구현하겠습니다.)

## 예상 소요 시간

5개 셀 전부 Step 2+3을 하면 총 4~6시간입니다. 보고서에 실제로 쓸 셀만 우선순위를
정해 진행하는 것을 권합니다 -- 예: 주 실험(100e_gemini, 100e_deepseek, bk21_deepseek)
먼저, 분기 보조 셀(nsa/kpass)은 나중.
