## A1 수기 검토 Step 1 -- nsa_deepseek 사실별 진단표

| fact_id | mention | with_qty | consistent | contradiction | unscored | contradiction_rate |
|---|---|---|---|---|---|---|
| initial_payment_or_denial_deadline_days | 668 | 65 | 19 | 46 | 603 | 71% |
| open_negotiation_period_days | 0 | 0 | 0 | 0 | 0 | - |
| idr_initiation_window_days | 0 | 0 | 0 | 0 | 0 | - |
| hhs_complaint_response_deadline_days | 0 | 0 | 0 | 0 | 0 | - |
| abd_appeal_window_days | 0 | 0 | 0 | 0 | 0 | - |
| erisa_extension_days | 0 | 0 | 0 | 0 | 0 | - |

**진단 플래그**

- `initial_payment_or_denial_deadline_days` [mention_high_with_qty_low]: mention=668, with_qty=65 -- 언급은 하나 근처(±60자)에 같은 단위 종류 수치가 없음. 별칭 보강으로는 안 고쳐짐; window 폭이나 단위필터 로직 자체를 재검토할 사안.
- `initial_payment_or_denial_deadline_days` [high_contradiction_rate]: 모순 비율 71% (46/65) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `open_negotiation_period_days` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `idr_initiation_window_days` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `hhs_complaint_response_deadline_days` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `abd_appeal_window_days` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `erisa_extension_days` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.