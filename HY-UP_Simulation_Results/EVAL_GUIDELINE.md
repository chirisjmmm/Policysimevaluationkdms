# 정책 시뮬레이션 발화 평가지표 구축 가이드라인 (v3)

LLM 다중에이전트 정책 시뮬레이션의 **발화 품질**을 Accuracy / Diversity 축으로 평가하는
로컬 파이프라인 구현 명세.

> **제약**: 외부 유료 API(Gemini / Claude / OpenAI) 사용 불가. 모든 지표는 결정적 로컬
> 연산 또는 로컬 캐시된 오픈 모델로만 계산한다. LLM-as-judge 금지.

## 이 문서의 기준 데이터

**주 대상 — 단일 pathway, 3개 셀 × 50 = 150 runs**

| 셀 ID | 정책 | 정책유형 | 모델 | runs |
|---|---|---|---|---|
| `100e_gemini` | 100 Experiments (싱가포르) | 분배 | Gemini | 50 |
| `100e_deepseek` | 100 Experiments (싱가포르) | 분배 | DeepSeek | 50 |
| `bk21_gemini` | Brain Korea 21 (한국) | 분배 | Gemini | 50 |

**보조 대상 — assumption-conditioned(분기) pathway**: NSA(규제형), K-패스(분배형).
각 정책당 트리 1개. 주 실험이 아니라 **해석기준 예비 검증용**으로만 쓴다.

## 지표 적용 가능성 표기

문서 전체에서 각 지표에 아래 태그를 붙인다.

| 태그 | 의미 |
|---|---|
| ✅ | 단일·분기 모두 계산 가능 |
| △ | 분기에서 계산은 되지만 **보정 또는 해석 주의** 필요 |
| ⛔ | **분기에서 계산 불가 또는 신뢰 불가** (단일 전용) |
| 🌿 | **분기 전용** (단일에서 계산 불가) |

---

## 0. 설계 원칙

1. **변수는 두 종류다.** 문서에서 베껴오는 `anchored_constant`와 실제로 예측하는
   `predicted`. 전자의 분산은 다양성이 아니라 **오류**이고, 후자의 분산이 다양성 신호다.
   이 구분 없이 계산하면 모든 지표가 오독된다. (§3.4)
2. **경계 인지 판정.** 정책 문서의 수치는 `up to X`(상한), `more than X`(하한),
   `6-12 months`(구간)로 표현된다. 점값 동치로 판정하면 정상 발화를 오류로 찍는다.
3. **예측값 ≠ 사실값.** 정책 상수를 지칭하는 문맥에서만 문서와 대조한다.
4. **셀을 가로질러 원자료를 풀링하지 않는다.** 모델 효과가 크다. (§6)
5. **Diversity 단독 보고 금지.** Accuracy와 Pareto로.
6. **모든 판정에 근거 span을 남긴다.**
7. **결정성.** 시드 고정, 부트스트랩 시드 명시.

---

## 1. 저장소 구조

```
policy-sim-eval/
├── configs/
│   ├── cells.yaml                 # 3개 셀 정의  ★신설
│   ├── policy_100e.yaml
│   ├── policy_bk21.yaml
│   ├── policy_nsa.yaml            # 보조 (분기)
│   ├── policy_kpass.yaml          # 보조 (분기)
│   └── eval_default.yaml
├── data/policies/{100e,bk21,nsa,kpass}/{doc.md, graph.json, runs/}
├── src/
│   ├── io/        loaders.py  ir.py  normalize.py  offsets.py
│   ├── prep/      doc_clean.py  kg_clean.py  var_typing.py
│   ├── index/     doc_index.py  kg_index.py  retriever.py
│   ├── metrics/   acc_micro.py  acc_meso.py  acc_macro.py
│   │              div_micro.py  div_meso.py  div_macro.py
│   ├── design/    cells.py  contrasts.py        # 셀 간 비교  ★신설
│   ├── report/    build_report.py  viz.py
│   └── cli.py
├── tests/
└── outputs/
```

**의존성** — 전부 무료·로컬

```
numpy scipy pandas scikit-learn networkx rapidfuzz pyyaml
# 선택: sentence-transformers (+ all-MiniLM-L6-v2). 로드 실패 시 TF-IDF 자동 폴백.
```

---

## 2. 데이터 계약 (실측)

### 2.1 단일 pathway 시나리오 — 주 대상

```jsonc
{
  "scenario_id": "...",
  "participants": [ {"persona_id","stakeholder_type","name","professional_persona",
                     "skills_and_expertise","career_goals_and_ambitions",
                     "institution":"", "role":"", "background":"",   // ← 항상 빈 문자열
                     "sex","age","education_level","occupation","district",
                     "bachelors_field"} ],                            // 5명
  "forward_pass": [
    {"phase","direction":"forward","posting_order":[...],             // ← phase마다 로테이션
     "initial_posts": [{"persona_name","stakeholder_type",            // ← persona_id 없음
                        "prediction_values":{...},"narrative","evidence":[...],
                        "judgment": null}],
     "revised_posts": [ /* initial과 전량 동일 */ ],
     "refined_posts": null, "phase_summary": "..."} ],
  "backward_pass": null, "fwd_bwd_fwd_pass": null, "fwd_bwd_fwd_bwd_pass": null,
  "cross_checks": [{"phase","variable","forward_agg","backward_agg":null,
                    "gap":null,"relative_gap":null,"consistency":null,
                    "aggregated_value"}],                             // = 5인 산술평균
  "scenario_impact": {...}, "scenario_confidence": {"<var>": null}
}
```

### 2.2 분기 pathway — 보조 대상

```jsonc
{ "format_version":1, "generation_mode":"phase_local_independent_children",
  "phase_interaction_mode":"independent", "branch_inputs": false,
  "phases":[5], "transitions":["enabling","baseline","constraining"],
  "simulation_target":"...", "completed_parent_ids":[40],
  "nodes": { "root/baseline/enabling/...": {
      "node_id","parent_id","transition_mode","phase_index","parent_context_hash",
      "phase_result": /* 2.1 forward_pass 원소와 동일 스키마 */ } } }   // 121 = 1+3+9+27+81
```

노드 키가 경로다. `key.count("/") == phase_index`. root→leaf 경로 81개,
Outcomes 노드 27개.

### 2.3 KG

```jsonc
{ "ontology": {"entity_types":[{"name","iad_category","is_actor","persona_eligible",
                               "attributes":[...]}],
                "edge_types":[...], "analysis_summary":{...}},
  "nodes": [{"id":"Type:Name","name","type","iad_category","summary","attributes":{...},
             "source_spans":[{"chunk_id","text"}]}],
  "edges": [{"id","type","edge_category":"structural|regulative|constitutive",
             "source","target","fact","deontic","activation_condition","source_text",
             "attributes":{"evidence_alignment":...}}],
  "metadata": {"chunk_count","chunk_size":1200,"chunk_overlap":200,...},
  "validation_warnings": [ ... ] }
```

100E 실측: nodes 218 / edges 280 / warnings 1,366 / 수치 속성 15개.

### 2.4 공통 IR

```python
@dataclass(frozen=True)
class Post:
    cell_id: str                     # "100e_gemini" 등
    run_id: str; path_key: str
    phase: str; phase_index: int
    order_index: int
    agent_key: str                   # persona_name
    stakeholder_type: str
    role_slot: str                   # "AIResearcher#1" — 동일 타입 중복 대응
    prediction_values: dict[str, float]
    narrative: str; evidence: list[str]
    transition_mode: str | None       # 분기 전용

@dataclass(frozen=True)
class Run:
    cell_id: str
    run_id: str
    source: Literal["single","branch"]
    transition_path: tuple[str, ...]
    phases: list[PhaseState]
    target_values: dict[str, float]   # GT 단계(Outcomes)의 값
```

**분기 → Run 변환**: 정확도용은 Outcomes 노드 27개, 다양성용은 leaf 81개.
표본 단위가 다르다는 점을 리포트에 명시한다.

---

## 3. 전처리 명세 ★

### 3.1 정책 문서 (`prep/doc_clean.py`)

| 문제 | 100E 실측 | 대응 |
|---|---|---|
| 페이지 마커 `--- page N ---` | 68개 | 페이지 번호를 메타로 보존, 본문에서 제거 |
| 이미지 플레이스홀더 | 19개 | 제거 |
| PDF 추출 다중 공백 | 224개 라인 | 축약. **오프셋 맵 필수** |
| 곡선 아포스트로피 | 51개 | ASCII 정규화 |
| 목차 잔여 숫자 | 7개 라인 | **목차 구간 통째로 제외.** 안 하면 수치 추출 오염 |
| 페이지 푸터 | 3개 | 제거 |

BK21 문서(한국어)는 추가로 확인할 것: 한글 스케일어(억/만/조), 표 형식 잔여물,
각주 번호가 본문 수치와 붙는 경우(`2,700명1)` 형태).

**오프셋 맵**: 정규화 후에도 원문 좌표로 하이라이트해야 한다.

```python
clean_text, offset_map = normalize_with_map(raw_text)
def to_raw(span): return (offset_map[span[0]], offset_map[span[1]-1] + 1)
```

지표는 clean_text 위에서 계산하고, `evidence[]` 기록 시에만 `to_raw()`로 되돌린다.

### 3.2 KG (`prep/kg_clean.py`)

| 문제 | 100E 실측 | 대응 |
|---|---|---|
| `summary` 근중복 문장 | 11 / 218 노드 | 문장 분할 후 `SequenceMatcher > 0.75` 제거. **미처리 시 grounding 유사도 부풀림** |
| 중복 엣지 | 16건 | 병합 |
| **source_spans가 원문 인용 아님** | 99개 중 문자열 일치 **7건** | 아래 별항 |
| chunk_id ↔ 문서 정렬 불가 | metadata 88 vs 예측 71 | 아래 별항 |
| **추론 기반 노드** | warnings 1,366건 | 아래 별항 |
| 수치 속성 희소 | 15개 | 사실 테이블은 **문서 우선** |
| 노드 id `Type:Name` | Enterprise 33 / PolicyRole 20 | 역할 매핑은 **타입 레벨 집계** |

**source_spans 처리 원칙**

```
1. 문서 인덱스는 KG와 무관하게 독립 구축 (chunk_id에 의존하지 않음)
2. KG span → 문서 청크 정렬은 TF-IDF top-1 fuzzy 매칭, 유사도를
   alignment_confidence로 기록 (0.5 미만은 "정렬 실패")
3. 시각화에서 KG 근거는 "KG 요약(추정 출처: p.21)"으로 라벨링해
   문서 원문 인용과 시각적으로 구분
```

**신뢰도 등급**

| tier | 조건 | 사실 대조(A1) | grounding(A2) |
|---|---|---|---|
| `T1_explicit` | 경고 없음 + `evidence_alignment == "supported"` | ✅ | ✅ |
| `T2_weak` | 경고 있으나 원문 근거 언급 | ❌ | ✅ |
| `T3_inferred` | "inferred from ontology / not explicit text" | ❌ | ⚠️ 별도 채널 |

`T3`를 사실 대조에 쓰면 **KG의 환각으로 발화를 판정**하게 된다.

### 3.3 시나리오

| 문제 | 실측 | 대응 |
|---|---|---|
| `institution`/`role`/`background` 공백 | 전원 | 페르소나 특성은 `professional_persona` 등 자유텍스트에서만 |
| post에 `persona_id` 없음 | — | `persona_name`으로 조인, 참가자 목록과 대조 검증 |
| 페르소나 이름 퇴화 | `An`, `A (9s3qtv)` | 이름 매칭 인용 탐지 **금지** (`An` ↔ 관사) |
| 동일 `stakeholder_type` 중복 | AIResearcher×2 등 | `role_slot = f"{type}#{순번}"` |
| `posting_order` | 단일: **phase마다 로테이션** / 분기: **121노드 고정** | §5.2 M3 참조 |
| `revised_posts` no-op | 전량 동일 | `initial_posts`만 사용, `assert` 후 메타 기록 |
| `judgment`/`refined_posts` null | 전량 | 관련 지표 계산 금지 |
| `backward_pass` 계열 null | 전량 | `cross_checks`의 `gap`/`consistency`도 null |
| `aggregated_value` | 5인 산술평균 | 재계산 후 일치 검증 |
| 분기 발화 중복 | 605개 중 고유 582 | 중복 해시 탐지 후 다양성 가중치 축소 |

### 3.4 변수 유형 분류 (`prep/var_typing.py`) ★ 가장 중요

100E 단일 시나리오 실측 (5 에이전트 기준 고유값 개수):

```
Inputs      total_funding:1  co_funding:1  gpu_hours:1  trainer_count:1     ← 전원 합의
Activities  outreach:2  matchmaking:4  workshops:2  onboarding:1
Outputs     approved:5  prototypes:5  trainees:1  enterprises:5             ← 최대 분산
Outcomes    approved_2023:5  deployed:5  practitioner_pool:1  usecases:1  adoption:1
Impact      rank:1  gdp_share:1  trust:3  retention:3                       ← 재수렴
```

```python
def classify_variable(phase, var, fact_table, doc_index) -> str:
    """
    'anchored_constant'   : 문서/KG(T1)에 대응 수치 존재 → 분산은 '오류'
    'predicted'           : 문서에 근거 없음, 미래 추정 → 분산은 '다양성'
    'unanchored_constant' : 근거 없는데 분산 0 → 기술 통계로만 기록 (지표 아님)
    """
```

판정: ① config의 `variable_anchors` 수기 선언 우선 → ② 문서/KG(T1) 별칭 자동 검색 →
③ 분산 0 + 근거 없음 → `unanchored_constant` 플래그.

| 변수 유형 | Accuracy | Diversity |
|---|---|---|
| `anchored_constant` | 값 대조 ✅ | **제외** |
| `predicted` | GT 있을 때만 | **계산 대상** |
| `unanchored_constant` | 대조 대상 아님 | 제외 |

> **실측 사례**: `available_gpu_hours: 10,000,000`, `trainer_personnel_count: 200`은
> 문서에 근거가 없는데 5명 만장일치다. = **합의된 환각(consensus hallucination)**.

### 3.5 수치 정규화 (`io/normalize.py`)

```python
@dataclass(frozen=True)
class Quantity:
    value: float
    kind: str                 # currency|count|percent|percentage_point|rate|rank|hours|years
    currency: str | None
    bound: str                # exact | at_most | at_least | range   ★필수
    range_hi: float | None
    approx: bool
    raw: str
    span: tuple[int,int]      # clean_text 오프셋
```

| 문서 표현 | bound | value | 판정 |
|---|---|---|---|
| `up to S$330,000 per project` | `at_most` | 330000 | 발화값 ≤ 330000 → SUPPORTED |
| `more than S$500 million` | `at_least` | 5e8 | 발화값 ≥ 5e8 → SUPPORTED |
| `typically 6-12 months` | `range` | 6 (hi=12) | 구간 포함 → SUPPORTED |
| `S$330,000` (수식어 없음) | `exact` | 330000 | 상대오차 ≤ 0.5% |

> 이 규칙이 없으면 발화의 `total_funding_sgd = 550,000,000`을 오류로 찍는다.
> 문서가 "more than S$500 million"이므로 550M은 **정상**이다.

```python
CURRENCY = {"S$":"SGD","SGD":"SGD","US$":"USD","$":"USD","₩":"KRW","원":"KRW"}
SCALE_EN = {"thousand":1e3,"k":1e3,"million":1e6,"m":1e6,"billion":1e9,"bn":1e9}
SCALE_KO = {"천":1e3,"만":1e4,"억":1e8,"조":1e12}      # BK21용 필수

def satisfies(observed, ref, rel_tol=0.005) -> bool:
    if observed.kind != ref.kind: return False
    if observed.kind == "currency" and observed.currency != ref.currency: return False
    tol = rel_tol * max(abs(observed.value), abs(ref.value), 1.0)
    if observed.approx or ref.approx: tol *= 4
    match ref.bound:
        case "exact":    return abs(observed.value - ref.value) <= tol
        case "at_most":  return observed.value <= ref.value + tol
        case "at_least": return observed.value >= ref.value - tol
        case "range":    return ref.value - tol <= observed.value <= ref.range_hi + tol
```

- **환율 변환 금지.** 통화가 다르면 `comparable=False`.
- **`%` vs `%p` 혼동 금지.**

**테스트 필수 케이스**

| 입력 | value | kind | bound |
|---|---|---|---|
| `up to S$330,000` | 330000 | currency/SGD | at_most |
| `more than S$500 million` | 5e8 | currency/SGD | at_least |
| `S$0.33 million` | 330000 | currency/SGD | exact |
| `2.6억원` | 2.6e8 | currency/KRW | exact |
| `260,000,000원` | 2.6e8 | currency/KRW | exact |
| `4단계 BK21 사업 7년간` | 7 | years | exact |
| `typically 6-12 months` | 6 (hi=12) | months | range |
| `8.5 percentage points` | 8.5 | percentage_point | exact |
| `8.5%` | 8.5 | percent | exact |
| `about 2,500` | 2500 (approx) | count | exact |

---

## 4. 참조 인덱스

**문서 인덱스** — clean_text를 문단 단위 청킹(KG chunk_id와 독립).
`DocChunk(id, page, text, quantities, raw_span)`.

**KG 인덱스**

```python
class KGIndex:
    fact_table: dict[str, list[Fact]]  # T1 노드의 수치 속성만
    text_pool:  list[TextRef]          # (source_id, tier, text)
    role_types: set[str]               # persona_eligible == True
```

`text_pool` = `summary`(중복 제거 후) + `edge.fact` + `edge.source_text`. tier별 분리 저장.

**역할 매핑** — `stakeholder_type`을 ontology의 `persona_eligible` 타입과 정확 일치 →
실패 시 `rapidfuzz` cutoff 85. KG 노드는 타입당 다수이므로 타입 레벨 집계.
미매칭율을 리포트에 명시.

---

## 5. 지표 명세

### 5.1 Accuracy — Micro

#### A1. Factual Contradiction ✅ ★ 주력 지표

**목표를 명확히 한다**: 문서에 없는 항목이 발화에 새로 등장하는 것은 **문제가 아니다**.
ToC 시뮬레이션은 애초에 문서에 없는 값을 추정하라고 요구한다.
잡아야 하는 것은 오직 **문서가 명시한 값과 정면으로 어긋나는 발화**다.
(문서: 예산 2.6억 → 발화: 예산 1.3억)

**탐지 방향이 중요하다.** 발화에서 출발하면 별칭 사전에 없는 표현을 쓴 모순을 놓친다.
**사실에서 출발**해야 재현율이 확보된다.

```python
# 사실 앵커 방향 (fact-anchored). 검사 대상이 20~40개로 한정되므로
# 별칭을 넉넉히 잡아 재현율을 올리고, 오탐은 근거 span 검토로 거른다.
for fact in fact_table:                       # 정책당 20~40개
    for post in all_posts:
        for m in find_alias_mentions(post.text, fact.aliases):   # 별칭 등장 위치
            for q in quantities_near(post.text, m.span, window=60):
                if q.kind != fact.kind: continue        # 단위 종류 불일치는 건너뜀
                if satisfies(q, fact):  -> CONSISTENT
                else:                   -> CONTRADICTION  # span + fact 근거 첨부
```

$$
A1 = 1 - \frac{|\text{CONTRADICTION}|}{|\text{CONSISTENT}| + |\text{CONTRADICTION}|}
$$

분모는 **그 사실을 언급한 발화**만이다. 언급이 없으면 분모에서 빠지고,
언급했는데 틀린 경우만 감점된다. 문서에 없는 새 항목은 애초에 검사 대상이 아니다.

**보조 보고**: 사실별 언급 빈도(`mention_count`)를 함께 낸다.
언급 빈도 0인 사실이 많으면 별칭이 부족하거나 시뮬레이션이 그 사실을 무시한 것이다.

**함정**: `kind` 불일치 필터를 반드시 넣는다. "예산"이라는 별칭 근처에
연도(2023)나 과제 수(125)가 있을 수 있는데, 이걸 금액과 대조하면 전부 모순으로 찍힌다.
단위 종류가 같은 수치만 후보로 삼는다.

**수작업**: 사실 테이블(정책당 20~40행). 반자동화 가능 — §7.1 참조.

#### A2. Grounding Profile ✅

evidence 각 문장 + narrative 문장 분할본을 4개 풀 중 argmax로 배정.

| 출처 | 풀 |
|---|---|
| `doc` | DocChunk.text |
| `kg` | KGIndex.text_pool (tier 기록) |
| `prior` | 같은 phase 선행 post + 직전 phase `phase_summary` |
| `none` | 전부 max sim < τ (기본 0.35, TF-IDF cosine) |

```python
GroundedTotal   = 1 - mean_phases(none_share)
TransitionScore = (spearman(phase_index, prior_share) + 1) / 2
AnchorRetention = min_phases(doc_share + kg_share)
```

`TransitionScore`는 **후반부로 갈수록 이전 발화 참조가 느는 것을 가점**한다.
기대 프로파일 대비 `ProfileFit = 기대 구간 내 셀 비율`도 보고.

```yaml
expected_grounding:
  Inputs:     {doc_kg: [0.55, 0.95], prior: [0.00, 0.30]}
  Activities: {doc_kg: [0.40, 0.85], prior: [0.10, 0.50]}
  Outputs:    {doc_kg: [0.30, 0.75], prior: [0.20, 0.60]}
  Outcomes:   {doc_kg: [0.20, 0.65], prior: [0.30, 0.70]}
  Impact:     {doc_kg: [0.15, 0.60], prior: [0.35, 0.80]}
```

> **A3(무근거 합의)는 지표에서 제외했다.**
> ToC 시뮬레이션은 문서에 없는 값을 추정하도록 설계된 구조이므로,
> 문서에 근거 없는 항목이 등장하는 것 자체는 실패가 아니다.
> 다만 **변수 유형 분류(§3.4)는 그대로 유지한다** — A3 때문이 아니라
> Diversity 계산에서 `anchored_constant`를 제외해야 하기 때문이다.
> (Inputs의 분산 0을 다양성 계산에 넣으면 "다양성 없음"으로 오집계된다)

---

### 5.2 Accuracy — Meso

계산 대상은 `predicted` 변수만.

#### M1. Convergence ✅

```
CV(z,v) = std / max(|mean|, eps)                    # 변동계수
Convergence(z) = 1 - clip(median_v CV(z,v) / CV_ref, 0, 1)     # CV_ref=0.5
ConvergenceTrend = spearman(phase_index, Convergence(z))
```

수준과 추세를 함께 본다. 처음부터 전원 동일이면 수렴이 아니라 상호작용 부재다.

> **참고 실측**: NSA(규제형, 분기) 전체 median CV = 0.118,
> 100E(분배형, 단일 1건) = 0.013. 유형 간 차이가 클 가능성. 3개 셀로 확인할 것.

#### M2. Value Uptake ✅

이름 매칭 대신 **수치 반향**으로 인용을 탐지한다.

```python
def value_uptake(phase) -> float:
    hits = trials = 0
    for k, post in enumerate(ordered(phase.posts)):
        if k == 0: continue
        prior = {v for p in ordered(phase.posts)[:k] for v in p.prediction_values.values()}
        mine  = set(post.prediction_values.values())
        seen  = {q.value for q in extract_quantities(post.narrative)}
        hits += len({v for v in seen & prior if v not in mine}) > 0
        trials += 1
    return hits / max(trials, 1)
```

보조로 이름 인용도 세되 필터 필수: 3자 미만 별칭 제거, 불용어 충돌 별칭 제거,
`\b` 경계 + 대소문자 구분, `X's` 소유격 우선.

#### M3. Anchoring 계수 ⛔ **단일 전용**

```
y_k = α + β · mean(y_1..y_{k-1}) + ε        # 변수별 z-score 후 OLS
M3 = 1 - |β - β*| / max(β*, 1-β*)           # β* 기본 0.4
```

β≈0은 병렬 독백, β≈1은 무비판적 동조.

> **분기에서 계산 금지.** 분기 트리는 `posting_order`가 121노드 전부 고정이라
> 순서효과와 역할효과가 완전히 교란된다. 단일 pathway는 phase마다 로테이션되므로
> 순서효과 식별이 가능하다 — **이 지표는 단일 pathway의 고유 강점**이다.

---

### 5.3 Accuracy — Macro

**전제: ground truth가 Outcomes 단계에만 존재한다.**

```
체인 종단(chain terminal) : Impact   ← GT 없음
평가 종단(eval terminal)  : Outcomes ← GT 존재  ★ 정확도는 여기서 끝난다
```

Impact는 정확도 평가 제외, **일관성·다양성 전용**으로 격하한다.
분기의 경우 정확도 단위는 leaf 81개가 아니라 **Outcomes 노드 27개**.

| 단계 | 대리 지표 |
|---|---|
| Inputs | **A1** — 문서 상수가 가장 많이 언급되는 단계. 여기가 주력 |
| Activities | X2만 |
| Outputs | X2 + **X3 개념 중복 전파** |
| Outcomes | 본 정확도 (X1, X4) |
| Impact | 정확도 없음. X5만 |

#### X1. Target Error ✅

```
AE = |ŷ - y_true|;  APE = AE / |y_true| * 100
```

100E와 BK21은 각각 타깃이 2개다(100E: 승인·시행 과제 수 / BK21: 졸업생 취업률·국제협력률).
셀별 부트스트랩 500회(시드 고정)로 CI를 붙이고 `APE [95% CI]` 형태로 보고.

#### X2. Chain Internal Consistency ✅ (GT 불필요)

```yaml
chain_constraints:
  - "Outputs.ai_prototypes_developed <= Outputs.approved_100e_projects"
  - "Outputs.enterprises_with_solution <= Outputs.approved_100e_projects"
  - "Outcomes.deployed_100e_projects <= Outputs.approved_100e_projects"
X2 = 1 - 위반 제약 / 전체 제약
```

`ast.parse`로 안전 평가(`eval` 금지). **제약 개수를 함께 보고** — 정책마다 도출 가능한
제약 수가 다르므로 X2만 비교하면 안 된다.

#### X3. Concept-Duplicate Propagation ✅ ★

`Outputs.approved_100e_projects`와 `Outcomes.number_of_100e_projects_approved_in_2023`은
**같은 개념**이다. Outcomes의 GT를 Outputs로 전파하면 **GT 지점이 하나 늘어난다.**

```yaml
concept_duplicates:
  - group: approved_projects_2023
    members: ["Outputs.approved_100e_projects",
              "Outcomes.number_of_100e_projects_approved_in_2023"]
    ground_truth_from: "Outcomes.number_of_100e_projects_approved_in_2023"
```

```
X3_internal = 그룹 내 값 일치도       # 자기모순 탐지
X3_upstream = 전파된 GT 대비 상류 APE  # 추가 정확도 지점
```

변수명 유사도 + 단위 일치로 후보를 자동 제안하고 사람이 승인한다.

#### X4. Calibration (Outcomes 한정) △

```python
coverage_50, coverage_90
PIT  = ecdf_over_runs(y_true)
CRPS = mean_i|y_i - y_true| - 0.5 * mean_{i,j}|y_i - y_j|
```

CRPS는 coverage와 sharpness를 동시에 벌한다.

> **분기 적용 시 △**: 27개 Outcomes 노드는 조상을 공유해 독립이 아니다.
> `n_eff = 27 / (1 + (m-1)·ICC)` 를 반드시 병기.

#### X5. Post-target Plausibility (Impact) ✅

정확도가 아니라 **타당성 점검**으로 명확히 라벨링하고 별도 블록에 배치.

```
- 값 범위 위반율 (rank ≥ 1, 비율 0~100 등)
- Outcomes → Impact 방향 일관성 (spearman)
- 산출 완료율
```

#### X6. Branch Label Fidelity 🌿 **분기 전용**

```
지표 방향 보정 후: enabling ≥ baseline ≥ constraining
X6 = 1 - 위반 형제쌍 / 전체 형제쌍
```

형제 삼중쌍 40세트(= `completed_parent_ids`)의 **짝지어진 설계**이므로
Friedman 검정 / Wilcoxon 부호순위로 검정 가능하다. 트리 1개로도 통계가 성립한다.

> ⚠️ NSA 실측에서 응급 OON 감소폭 중앙값이 enabling 8.0 / baseline 8.5 /
> **constraining 11.5** 로 라벨과 부호가 반대다. 지표 방향 정의와 프롬프트의
> enabling/constraining 의미를 **구현 전에** 확정할 것.

---

### 5.4 Diversity — Micro

계산 대상은 `predicted` 변수만.

**분산분해로 층위를 정의한다.**

```
Var_total = Var_within_role_across_runs    (micro)
          + Var_between_role_within_run    (역할 분화 — structural)
          + Var_between_run                (macro)

ICC_role = Var_between_role / Var_total
ICC_run  = Var_between_run  / Var_total
```

`scipy` + 수동 제곱합으로 구현. 동일 타입 중복은 `role_slot`으로 구분.

- **D1 Response Variance** ⛔ **단일 전용** — `Var_within_role_across_runs`.
  분기 파일은 페르소나 조합이 하나뿐이라(`S1_...` 고정) **이 성분이 구조적으로 0**이다.
  분기 데이터로 Diversity-micro는 계산 불가.
- **D2 Semantic Dispersion** ✅ — TF-IDF 코사인 pairwise 거리 평균 + distinct-2/3.
  임베딩 백엔드가 있으면 병행하되 **TF-IDF 결과를 항상 함께 보고**.
- **D3 Accuracy–Diversity Pareto** ✅ — D1/D2 vs X1/A1 산점도.

> **참고 실측**: NSA ICC_role 중앙값 = 0.042. 역할 간 분산 비중이 매우 낮다.
> 페르소나가 이름만 다르고 관점이 역할에 결속되지 않았을 가능성 — 3개 셀에서
> 같은 값을 계산해 비교할 것.

---

### 5.5 Diversity — Meso

**점수는 2개만.** 나머지는 진단용 서술 통계로 부록에만 싣는다.

**공통 전처리 — 영향 그래프**

```python
nodes = 5 agents
edge i -> j  (i가 j보다 먼저 발화) weight =
      0.5 * 1[value_uptake(j from i)]
    + 0.2 * 1[name_citation(j → i)]
    + 0.3 * shrinkage(j toward i)
shrinkage(j,i) = max(0, |ŷ_j^base - y_i| - |y_j - y_i|) / scale
```

#### DM1. Interaction Structure Diversity △ — **핵심 지표**

| 유형 | 판정 규칙 |
|---|---|
| `disconnected` | 총 엣지 가중치 < 0.15 |
| `star` | 최대 in-degree / 평균 ≥ 2.0, 앵커 1명이 유입 50%↑ |
| `chain` | 엣지 70%↑가 직전 발화자에서 옴 |
| `complete` | 위 해당 없고 밀도 ≥ 0.5 |

```
DM1 = H(유형 분포 over runs) / log(4)
```

부속 서술 통계(점수 아님): 앵커 역할의 `stakeholder_type` 분포, 유형별 빈도표.

> **분기 적용 시 △**: `posting_order` 고정 때문에 `chain` 유형이 과대 추정된다.
> 분기에서는 `chain`을 별도 표시하고 단일과 직접 비교하지 말 것.

#### DM2. Coalition Variability ✅ — **보조 지표**

phase별 값 거리 행렬 → `scipy.cluster.hierarchy` 파티션 →
run 쌍마다 `adjusted_rand_score`.

```
DM2 = 1 - mean_pairwise_ARI       # 낮은 ARI = 구조적으로 다양
```

연합의 **구성**(어느 역할끼리 묶이는가)도 기록. Lowi 해석기준에서 부호가 뒤집히는 지표다.

#### DM3. Sibling Structural Divergence 🌿 **분기 전용**

```
DM3 = struct_div / max(value_div, eps)
```

낮으면 **"가정 조건이 숫자만 흔들고 심의 구조는 못 바꿨다"**는 진단.

> **선결 조건**: `phase_interaction_mode: "independent"`이므로 에이전트 i의 프롬프트가
> 실제로 1..i-1의 발화를 보았는지 로그로 확정할 것. 안 봤다면 meso 점수가 낮은 것은
> 모델 실패가 아니라 **설계의 결과**다.

---

### 5.6 Diversity — Macro

- **DX1 Outcome Support Width** ✅ — Outcomes 값의 range / IQR / std
- **DX2 Bin Distribution Entropy** △ — GT 기준 상대구간 binning 후 엔트로피.
  **표본 수가 다르면 rarefaction 필수** (단일 3셀은 전부 n=50이라 셀 간에는 불필요,
  분기 27/81과 비교할 때만 필요)
- **DX3 Trajectory Clustering** ✅ — run × (phase × predicted 변수) → z-score → PCA →
  KMeans (k는 silhouette 2~8 탐색, 시드 고정). 클러스터 중심 = "대표 정책 시나리오"
- **DX4 분기 기여도 분해** 🌿 — 어느 phase의 분기가 최종 분산을 만드는가

---

## 6. 실험 설계와 셀 간 비교 (`src/design/`)

### 6.1 셀 구성

```yaml
# configs/cells.yaml
cells:
  - {id: 100e_gemini,   policy: 100e, policy_type: distributive, model: gemini,   n: 50}
  - {id: 100e_deepseek, policy: 100e, policy_type: distributive, model: deepseek, n: 50}
  - {id: bk21_gemini,   policy: bk21, policy_type: distributive, model: gemini,   n: 50}
```

**불완전 2×2 설계**다. 추정 가능한 대비와 불가능한 대비를 명확히 구분한다.

| 대비 | 방법 | 가능? |
|---|---|---|
| **모델 효과** | `100e_gemini` vs `100e_deepseek` (정책 고정) | ✅ |
| **정책 효과** | `100e_gemini` vs `bk21_gemini` (모델 고정) | ✅ |
| **상호작용** | 정책 × 모델 | ❌ `bk21_deepseek` 부재 |

> `bk21_deepseek` 50개를 추가하면 2×2가 완성되고 상호작용을 추정할 수 있다.
> TIPS 논문 Table 1에서 모델별로 정책 순위가 뒤바뀌는 셀이 있으므로
> 상호작용은 실재할 가능성이 있다. **우선순위 높음.**

### 6.2 집계 규칙

```
1. 셀을 가로질러 원자료를 풀링하지 않는다.
   → 특히 100e_gemini와 100e_deepseek을 "100E"로 합치지 말 것.
2. 주 결과표는 셀별 행으로 제시한다.
3. 요약이 필요하면 셀 평균의 비가중 평균(주변평균)을 쓴다.
4. 모든 셀 통계에 부트스트랩 500회 CI를 붙인다 (시드 고정).
5. 세 셀 모두 n=50으로 균형 → 엔트로피 계열 rarefaction 불필요.
   단, 분기(27/81)와 비교할 때는 최소 n으로 반복 서브샘플링.
```

### 6.3 안정성 확인

TIPS 논문은 시나리오 약 50개에서 추정이 안정화된다고 보고한다.
현재 셀당 n=50은 **그 임계점에 정확히 걸쳐 있다.** 셀마다
`N = 10, 20, 30, 40, 50` 서브샘플 학습곡선(부트스트랩 500회, ±1σ 밴드)을 그려
안정화 여부를 직접 보이고, 밴드가 여전히 넓으면 리포트에 명시한다.

### 6.4 보조 대상(분기) 처리

```
1. 정확도 표본 단위 = Outcomes 노드 27개, 다양성 = leaf 81개 (서로 다름을 명시)
2. n_eff = 27 / (1 + (m-1)·ICC) 병기
3. 페르소나 조합이 1개뿐 → D1(Diversity-micro) 계산 불가로 표기
4. posting_order 고정 → M3 계산 금지, DM1의 chain 유형 별도 표시
5. NSA vs K-패스는 경로 구조·모델(deepseek)이 통제된 비교 → 정책 유형 대비로 활용
```

---

## 7. 설정 파일

### 7.1 사실 테이블 만들기 (유일한 수작업, 정책당 30~60분)

A1의 재현율은 전적으로 이 테이블에 달려 있다. 다만 **완전 수작업은 아니고 2단계 반자동**이다.

**1단계 — 후보 자동 추출 (코드가 함)**

```
전처리된 문서에서 수치를 포함한 모든 문장을 추출해
fact_candidates.csv 로 저장한다. 컬럼:
  value | kind | currency | bound | context_40 | page | raw_sentence
판단은 하지 않고 후보만 나열한다.
```

100E 기준 30~50행 정도가 나온다. `bound`는 `up to` / `more than` / `at least` /
`최대` / `이상` / `범위(A-B)` 패턴으로 자동 태깅한다.

**2단계 — 사람이 선별 (30~60분)**

각 후보에 대해 세 가지만 판단한다.

1. **검증 대상인가?** — 정책이 규정한 상수(예산 상한, 사업 기간, 정원)면 O,
   지나가는 예시나 배경 통계면 X
2. **별칭을 무엇으로 잡을까?** — 에이전트가 이 사실을 언급할 때 쓸 법한 표현을
   **넉넉하게** 나열한다. 재현율이 목표이므로 오탐을 두려워하지 말 것
3. **경계 표현이 맞게 태깅됐나?** — `up to` / `more than` 자동 태깅 검수

```yaml
fact_table:
  - id: co_funding_cap
    value: 330000
    kind: currency
    currency: SGD
    bound: at_most
    aliases: ["co-funding", "co funding", "per project", "S$330,000", "330,000",
              "funding cap", "지원 한도"]
    evidence: "provides co-funding of up to S$330,000 per project"
    page: 30
  - id: total_rd_commitment
    value: 5.0e8
    kind: currency
    currency: SGD
    bound: at_least
    aliases: ["R&D funding", "R&D investment", "total funding",
              "national AI funding", "committed", "S$500 million"]
    evidence: "committed more than S$500 million through AI Singapore"
    page: 21
```

**BK21은 추가 검수 항목이 있다**: 한글 스케일어(억/만/조) 파싱,
각주 번호가 수치에 붙는 경우(`2,700명1)`), 표에서 추출된 수치의 헤더 소실.

### 7.2 정책 설정 파일

```yaml
# configs/policy_100e.yaml
policy_id: 100e
policy_type: distributive
doc_path: data/policies/100e/doc.md
graph_path: data/policies/100e/graph.json
runs_glob: data/policies/100e/runs/{model}/scenario_*.json
run_source: single

phases: [Inputs, Activities, Outputs, Outcomes, Impact]
eval_terminal_phase: Outcomes
post_target_phases: [Impact]

targets:
  - {name: number_of_100e_projects_approved_in_2023, phase: Outcomes,
     true_value: <official>, source: "AI Singapore 5-year report (2024)"}
  - {name: deployed_100e_projects, phase: Outcomes,
     true_value: <official>, source: "AI Singapore 5-year report (2024)"}

concept_duplicates:
  - group: approved_projects_2023
    members: ["Outputs.approved_100e_projects",
              "Outcomes.number_of_100e_projects_approved_in_2023"]
    ground_truth_from: "Outcomes.number_of_100e_projects_approved_in_2023"

variable_anchors:
  Inputs.max_co_funding_per_project_sgd:
    type: anchored_constant
    ref: {value: 330000, kind: currency, currency: SGD, bound: at_most}
    evidence: "provides co-funding of up to S$330,000 per project"
  Inputs.total_funding_sgd:
    type: anchored_constant
    ref: {value: 5.0e8, kind: currency, currency: SGD, bound: at_least}
    evidence: "committed more than S$500 million through AI Singapore"
  Inputs.available_gpu_hours:     {type: unanchored_constant}   # 문서 근거 없음
  Inputs.trainer_personnel_count: {type: unanchored_constant}   # 문서 근거 없음
  Outputs.approved_100e_projects: {type: predicted}

fact_aliases:
  max_co_funding_per_project_sgd: ["co-funding", "per project", "S$330,000", "funding cap"]
  total_funding_sgd: ["R&D funding", "total funding", "national AI funding", "committed"]

chain_constraints:
  - "Outputs.ai_prototypes_developed <= Outputs.approved_100e_projects"
  - "Outputs.enterprises_with_solution <= Outputs.approved_100e_projects"
  - "Outcomes.deployed_100e_projects <= Outputs.approved_100e_projects"

value_ranges:
  Impact.global_ai_readiness_rank: {min: 1, max: 200, integer: true}
  Impact.ai_sector_gdp_share: {min: 0, max: 100}
```

```yaml
# configs/policy_bk21.yaml  (골격 — 실제 값은 문서 확인 후 채울 것)
policy_id: bk21
policy_type: distributive
eval_terminal_phase: Outcomes
targets:
  - {name: graduate_employment_rate, phase: Outcomes, true_value: <official>,
     source: "2022 4단계 BK21 사업 관리·운영 계획 (교육부)"}
  - {name: international_collaboration_rate, phase: Outcomes, true_value: <official>,
     source: "동일"}
# 한국어 문서이므로 SCALE_KO(억/만/조) 파싱과 각주 번호 분리 검증 필수
```

```yaml
# configs/eval_default.yaml
seed: 20260810
retriever: {backend: auto, tau_grounding: 0.35}
normalize: {rel_tol: 0.005, approx_tol_multiplier: 4}
kg: {use_tiers_for_facts: [T1_explicit], use_tiers_for_grounding: [T1_explicit, T2_weak]}
meso: {beta_star: 0.4, w_uptake: 0.5, w_citation: 0.2, w_shrink: 0.3,
       theta_disconnected: 0.15, theta_complete: 0.5}
diversity: {kmeans_k_range: [2, 8], cv_ref: 0.5}
bootstrap: {n_resamples: 500}
stability_curve: {n_points: [10, 20, 30, 40, 50]}
```

---

## 8. 출력 스키마 (`outputs/evaluation.json`)

```jsonc
{
  "meta": {"cells": ["100e_gemini","100e_deepseek","bk21_gemini"],
           "eval_terminal_phase":"Outcomes","post_target_phases":["Impact"],
           "retriever_backend","seed","generated_at",
           "revise_noop_verified": true, "aggregated_value_matches_mean": true,
           "kg_tier_counts": {"T1":..,"T2":..,"T3":..},
           "kg_span_literal_match_rate": 0.07,
           "chunk_alignment_confidence_median": 0.42,
           "variable_typing": {"anchored":..,"predicted":..,"unanchored":..}},

  "by_cell": {
    "100e_gemini": {
      "n_runs": 50,
      "accuracy": {
        "micro": {"A1": {"supported":..,"contradicted":..,"unscored":..,"score":..},
                  "A2": {"GroundedTotal":..,"TransitionScore":..,"AnchorRetention":..,
                         "ProfileFit":.., "by_phase": {...}},
                  "A1_by_fact": {"<fact_id>": {"mentions":..,"contradictions":..}}},
        "meso":  {"M1":..,"M1_trend":..,"M2":..,"M3_beta":..},
        "macro": {"X1": {"<target>": {"APE":..,"CI95":[..,..]}},
                  "X2": {"score":..,"n_constraints":..},
                  "X3": {"internal":..,"upstream_APE":..},
                  "X4": {"coverage_50":..,"coverage_90":..,"CRPS":..}}},
      "post_target": {"X5": {...}},
      "diversity": {"micro": {"D1":..,"D2":..,
                              "variance_decomposition": {"within_role":..,
                                "between_role":..,"between_run":..},
                              "ICC_role":.., "ICC_run":..},
                    "meso":  {"DM1":..,"DM2":..,
                              "descriptive": {"topology_counts":{...},
                                              "anchor_role_dist":{...}}},
                    "macro": {"DX1":..,"DX2":..,"DX3":{"k":..,"entropy":..}}},
      "stability_curve": {"10":..,"20":..,"30":..,"40":..,"50":..}
    }
  },

  "contrasts": {
    "model_effect_100e":  {"metric":"...", "gemini":.., "deepseek":.., "delta_CI":[..,..]},
    "policy_effect_gemini": {"metric":"...", "100e":.., "bk21":.., "delta_CI":[..,..]},
    "interaction": null                       // bk21_deepseek 부재
  },

  "branch_supplementary": {
    "nsa":   {"X6":.., "friedman_p":.., "M1":.., "ICC_role":.., "n_eff":..,
              "unavailable": ["M3","D1"]},
    "kpass": {"X6":.., "friedman_p":.., "M1":.., "ICC_role":.., "n_eff":..,
              "unavailable": ["M3","D1"]}
  },

  "evidence": [
    {"id":"ev_0001","metric":"A1","verdict":"SUPPORTED",
     "cell_id":"100e_gemini","run_id":"...","phase":"Inputs",
     "agent_key":"Ravi","role_slot":"PolicyRole#1",
     "utterance_span_raw":[4120,4142],"utterance_text":"S$550 million in R&D funding",
     "parsed":{"value":5.5e8,"kind":"currency","currency":"SGD","bound":"exact"},
     "reference":{"source":"doc","page":21,"bound":"at_least","value":5.0e8,
                  "text":"committed more than S$500 million through AI Singapore"},
     "note":"하한 조건 충족"}
  ],
  "influence_graphs": {"<cell>::<run_id>::<phase>": {...}},
  "distributions": {"<cell>::<target>": {"values":[...], "true_value":...}}
}
```

---

## 9. 구현 순서

| 단계 | 산출물 | 이유 |
|---|---|---|
| 1 | `normalize.py` (경계 인지, 한글 스케일 포함) + 테스트 10케이스 | 정확도 전체가 의존 |
| 2 | `doc_clean.py` + 오프셋 맵 (100E·BK21 각각 검증) | 목차 노이즈 미제거 시 수치 추출 오염 |
| 3 | `loaders.py` → IR, cell_id 부여, revise/aggregated_value assert | 3셀 통합 전제 |
| 4 | `var_typing.py` — 변수 유형 분류 | **모든 다양성 지표의 전제** |
| 5 | `kg_clean.py` — tier 부여, summary 중복 제거 | T3를 사실 대조에 쓰면 안 됨 |
| 6 | **A1 (사실 앵커 방향)** + evidence 로그 | 포스터의 핵심 발견. 가장 먼저 결과가 나옴 |
| 7 | M1 + 단계별 CV 분포 | 포스터 주 그림. 셀 간 비교 즉시 가능 |
| 8 | X2 + X3 | GT 지점을 늘리는 유일한 방법 |
| 9 | A2 grounding 프로파일 | |
| 10 | `design/contrasts.py` — 모델 효과·정책 효과 대비 | 주 실험 결과표 |
| 11 | D1/D2 + 분산분해 (ICC_role) | Diversity 축 |
| 12 | DM1 → DM2 | |
| 13 | 분기 보조: X6 + Friedman 검정 | Additional 칸 |
| 14 | `viz.py` 정적 HTML | 마지막 |

> **포스터 마감이 급하면 6 → 7 → 13 순으로만 먼저 만든다.**
> A1(문서 모순), 단계별 CV, X6(라벨 부호 역전) 셋이면 세 패널이 다 채워진다.

---

## 10. 함정 체크리스트

**전처리**
- [ ] 목차 구간 미제거 → 수치 추출 오염
- [ ] 공백 축약 후 오프셋 맵 미보존 → 시각화 하이라이트 어긋남
- [ ] BK21 한국어 문서에 SCALE_KO 미적용 → `2.6억원` 파싱 실패
- [ ] KG `source_spans.text`를 원문 인용으로 사용 → 실제 일치율 7%
- [ ] `validation_warnings`의 T3(추론 노드)를 사실 대조에 사용
- [ ] KG `summary` 근중복 문장 미제거 → grounding 유사도 부풀림
- [ ] KG `chunk_id`로 문서 정렬 시도 → 정렬 불가

**판정 로직**
- [ ] 경계 표현 무시 → `up to`/`more than`을 점값으로 보면 정상 발화를 오류 판정
- [ ] 통화 환산 / `%` vs `%p` 혼동
- [ ] `anchored_constant`를 다양성 계산에 포함 → Inputs 다양성 0을 실패로 오독
- [ ] 예측값을 문서와 대조해 감점

**설계·집계**
- [ ] `100e_gemini` + `100e_deepseek`를 "100E"로 풀링 → 모델 효과가 큼
- [ ] 상호작용을 주장 → `bk21_deepseek` 부재로 추정 불가
- [ ] n=50이 안정화 임계점에 걸쳐 있음 → 학습곡선으로 직접 확인
- [ ] 단일 타깃 APE 점추정만 보고 → 부트스트랩 CI 필수
- [ ] Diversity 단독 순위표 → Pareto로

**분기 보조 대상**
- [ ] 분기에서 M3 계산 → `posting_order` 고정으로 교란 (⛔)
- [ ] 분기에서 D1 계산 → 페르소나 조합 1개뿐이라 구조적으로 0 (⛔)
- [ ] 분기 Outcomes 27개를 독립 표본 취급 → n_eff 병기 (△)
- [ ] 분기 DM1의 `chain` 비율을 단일과 직접 비교 (△)
- [ ] X6 라벨 부호 역전 미확인 상태로 해석 → NSA에서 constraining이 더 큰 개선

**데이터 특성**
- [ ] `persona_name` 문자열 매칭으로 인용 탐지 (`An` ↔ 관사)
- [ ] 동일 `stakeholder_type` 중복 무시 → `role_slot` 필요
- [ ] `revised_posts`/`judgment`/`backward_pass` null 계열로 지표 계산 시도
