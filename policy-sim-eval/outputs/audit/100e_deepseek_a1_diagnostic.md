## A1 수기 검토 Step 1 -- 100e_deepseek 사실별 진단표

| fact_id | mention | with_qty | consistent | contradiction | unscored | contradiction_rate |
|---|---|---|---|---|---|---|
| total_rd_commitment | 716 | 716 | 290 | 426 | 0 | 59% |
| co_funding_cap | 649 | 607 | 211 | 396 | 42 | 65% |
| 100e_project_duration | 17 | 17 | 17 | 0 | 0 | 0% |
| global_ai_index_top10 | 5 | 0 | 0 | 0 | 5 | - |
| existing_ai_rd_teams | 3 | 3 | 2 | 1 | 0 | 33% |
| existing_ai_research_faculty | 0 | 0 | 0 | 0 | 0 | - |
| existing_ai_startups | 0 | 0 | 0 | 0 | 0 | - |
| stanford_ai_vibrancy_rank | 0 | 0 | 0 | 0 | 0 | - |
| oxford_govt_ai_rank | 0 | 0 | 0 | 0 | 0 | - |
| tortoise_ai_index_rank | 0 | 0 | 0 | 0 | 0 | - |
| pet_sandbox_launch_year | 0 | 0 | 0 | 0 | 0 | - |
| ai_verify_foundation_launch_year | 0 | 0 | 0 | 0 | 0 | - |

**진단 플래그**

- `total_rd_commitment` [high_contradiction_rate]: 모순 비율 59% (426/716) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `co_funding_cap` [high_contradiction_rate]: 모순 비율 65% (396/607) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `global_ai_index_top10` [mention_high_with_qty_low]: mention=5, with_qty=0 -- 언급은 하나 근처(±60자)에 같은 단위 종류 수치가 없음. 별칭 보강으로는 안 고쳐짐; window 폭이나 단위필터 로직 자체를 재검토할 사안.
- `existing_ai_research_faculty` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `existing_ai_startups` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `stanford_ai_vibrancy_rank` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `oxford_govt_ai_rank` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `tortoise_ai_index_rank` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `pet_sandbox_launch_year` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `ai_verify_foundation_launch_year` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.