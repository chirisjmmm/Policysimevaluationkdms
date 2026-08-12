## A1 수기 검토 Step 1 -- kpass_deepseek 사실별 진단표

| fact_id | mention | with_qty | consistent | contradiction | unscored | contradiction_rate |
|---|---|---|---|---|---|---|
| refund_rate_low_income | 1372 | 1372 | 920 | 452 | 0 | 33% |
| min_monthly_usage_threshold | 1199 | 1199 | 1116 | 83 | 0 | 7% |
| refund_rate_youth | 265 | 265 | 0 | 265 | 0 | 100% |
| refund_rate_general | 188 | 188 | 10 | 178 | 0 | 95% |
| max_eligible_uses_per_month | 124 | 124 | 91 | 33 | 0 | 27% |
| card_issuer_count_at_launch | 0 | 0 | 0 | 0 | 0 | - |
| participating_metro_provinces | 0 | 0 | 0 | 0 | 0 | - |
| participating_municipalities | 0 | 0 | 0 | 0 | 0 | - |
| full_refund_spending_cap_krw | 0 | 0 | 0 | 0 | 0 | - |
| minimum_age | 0 | 0 | 0 | 0 | 0 | - |

**진단 플래그**

- `refund_rate_youth` [high_contradiction_rate]: 모순 비율 100% (265/265) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `refund_rate_general` [high_contradiction_rate]: 모순 비율 95% (178/188) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `card_issuer_count_at_launch` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `participating_metro_provinces` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `participating_municipalities` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `full_refund_spending_cap_krw` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `minimum_age` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.