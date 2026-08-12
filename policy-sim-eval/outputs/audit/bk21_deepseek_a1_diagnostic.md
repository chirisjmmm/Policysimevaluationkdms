## A1 수기 검토 Step 1 -- bk21_deepseek 사실별 진단표

| fact_id | mention | with_qty | consistent | contradiction | unscored | contradiction_rate |
|---|---|---|---|---|---|---|
| annual_budget_2022 | 324 | 5 | 1 | 4 | 319 | 80% |
| participants_future_talent_annual | 99 | 99 | 14 | 85 | 0 | 86% |
| employment_rate_2022_target | 60 | 52 | 0 | 52 | 8 | 100% |
| education_research_groups_innovation_talent | 45 | 45 | 24 | 21 | 0 | 47% |
| early_career_researcher_salary_min_monthly | 31 | 10 | 10 | 0 | 21 | 0% |
| innovation_support_universities | 25 | 25 | 0 | 25 | 0 | 100% |
| participants_innovation_talent_annual | 14 | 14 | 4 | 10 | 0 | 71% |
| operating_budget_cap_pct | 5 | 5 | 5 | 0 | 0 | 0% |
| masters_scholarship_min_monthly | 4 | 4 | 4 | 0 | 0 | 0% |
| phd_scholarship_min_monthly | 4 | 4 | 4 | 0 | 0 | 0% |
| phd_completion_scholarship_min_monthly | 1 | 1 | 1 | 0 | 0 | 0% |
| faculty_lecture_ratio_2022_target | 1 | 1 | 0 | 1 | 0 | 100% |
| total_project_duration_years | 0 | 0 | 0 | 0 | 0 | - |
| education_research_groups_future_talent | 0 | 0 | 0 | 0 | 0 | - |
| indirect_cost_cap_pct | 0 | 0 | 0 | 0 | 0 | - |

**진단 플래그**

- `annual_budget_2022` [mention_high_with_qty_low]: mention=324, with_qty=5 -- 언급은 하나 근처(±60자)에 같은 단위 종류 수치가 없음. 별칭 보강으로는 안 고쳐짐; window 폭이나 단위필터 로직 자체를 재검토할 사안.
- `annual_budget_2022` [high_contradiction_rate]: 모순 비율 80% (4/5) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `participants_future_talent_annual` [high_contradiction_rate]: 모순 비율 86% (85/99) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `employment_rate_2022_target` [high_contradiction_rate]: 모순 비율 100% (52/52) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `early_career_researcher_salary_min_monthly` [mention_high_with_qty_low]: mention=31, with_qty=10 -- 언급은 하나 근처(±60자)에 같은 단위 종류 수치가 없음. 별칭 보강으로는 안 고쳐짐; window 폭이나 단위필터 로직 자체를 재검토할 사안.
- `innovation_support_universities` [high_contradiction_rate]: 모순 비율 100% (25/25) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `participants_innovation_talent_annual` [high_contradiction_rate]: 모순 비율 71% (10/14) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `phd_completion_scholarship_min_monthly` [mention_near_zero]: mention=1 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `faculty_lecture_ratio_2022_target` [mention_near_zero]: mention=1 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `faculty_lecture_ratio_2022_target` [high_contradiction_rate]: 모순 비율 100% (1/1) -- 스코프 파생 또는 매칭 오류 의심. Step 3 표본 라벨링에서 이 fact_id 사례를 우선 확인할 것.
- `total_project_duration_years` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `education_research_groups_future_talent` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.
- `indirect_cost_cap_pct` [mention_near_zero]: mention=0 -- 별칭이 실제 발화 표현과 불일치할 가능성. Step 2(별칭 보강)에서 이 사실을 다루는 발화 몇 개를 열어 실제 표현을 확인할 것.