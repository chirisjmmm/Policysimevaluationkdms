# 정책 시뮬레이션 발화 평가지표 구현 가이드라인 (v4 · 최종)

LLM 다중에이전트 정책 시뮬레이션의 발화 품질을 Accuracy / Diversity 축으로 평가한다.

> **제약**: 외부 유료 API 사용 불가. 결정적 로컬 연산 또는 로컬 캐시된 오픈 모델만 사용.
> LLM-as-judge 금지.

## v4 변경 요약

- 지표를 **10개로 확정**. 2축 × 3층 6칸을 모두 채운다
- **M3(앵커링 계수) 삭제** — 정규화 인공물 + 발화 1회 구조에서 해석 불가
- **X2·X3_internal·X5 → 게이트로 격하** (점수 아님, 통과/실패 한 줄)
- **X3_upstream → X1에 흡수** (계산식 동일)
- **D1 재정의**: 원분산 → `ICC_role`(역할 설명 분산 비율)
- **DX1 신설** — Diversity-macro가 비어 있었음
- **A1 수기 검토 프로토콜 명문화** (§6)
- **분기 pathway 확장 절 신설** (§8)

---

## 0. 최종 지표 구성

| | micro | meso | macro |
|---|---|---|---|
| **Accuracy** | `A1` 문서 모순<br>`A2` 근거 보유율 | `M1` 수렴도<br>`M2` 값 인용률 | `X1` 예측 오차<br>`X4` 캘리브레이션 |
| **Diversity** | `D1` 역할 설명 분산<br>`D2` 의미 분산 | `DM1` 상호작용 구조 다양성 | `DX1` 결과 공간 폭 |

**게이트 3종** (점수가 아니라 통과/실패로 한 줄 보고)

| 코드 | 내용 |
|---|---|
| `G1` | 체인 일관성 — 단계 간 산술 제약 + 개념 중복 변수 간 값 일치 |
| `G2` | 값 범위 타당성 — 비율 변수 [0,100], 순위 ≥ 1 등 |
| `G3` | 산출 완료율 — 파싱 실패·null 비율 |

**분기 pathway 전용 1종**

| 코드 | 내용 |
|---|---|
| `B1` | 분기 라벨 충실도 — `enabling ≥ baseline ≥ constraining` + Friedman 검정 |

### 삭제한 지표와 사유 (한 줄)

| 삭제 | 사유 |
|---|---|
| `M3` 앵커링 계수 | 그룹 내 z-score가 음의 상관을 강제하는 인공물. 발화 1회 구조라 해석 대상도 빈약 |
| `A2.AnchorRetention` | ToC 구조상 Impact는 문서가 아닌 직전 단계에 근거하는 게 정상. 부당한 요구 |
| `A2.ProfileFit` | 근거 보유율과 중복. 기대 구간 설정이 임의적 |
| `A2.TransitionScore` | 5개 점의 순위상관이라 검정력 없음 |
| `X3_internal` | 정확도가 아니라 일관성 → `G1`로 이동 |
| `DM2` 연합 가변성 | 분배형 데이터만 있어 해석 대비 불가 → 부록 |
| `distinct-2/3` | `D2`와 동일 개념 → 부록 |

---

## 1. 저장소 구조와 의존성

```
policy-sim-eval/
├── configs/
│   ├── cells.yaml
│   ├── policy_{100e,bk21,nsa,kpass}.yaml
│   └── eval_default.yaml
├── data/policies/{정책}/{doc.md, graph.json, runs/, branches/}
├── src/
│   ├── io/        loaders.py  ir.py  normalize.py  offsets.py
│   ├── prep/      doc_clean.py  kg_clean.py  var_typing.py
│   ├── index/     doc_index.py  kg_index.py  retriever.py
│   ├── metrics/   acc_micro.py  acc_meso.py  acc_macro.py
│   │              div_micro.py  div_meso.py  div_macro.py  gates.py
│   ├── branch/    tree_loader.py  paths.py  label_fidelity.py   ★신설
│   ├── audit/     a1_diagnostic.py  a1_sampler.py               ★신설
│   ├── design/    cells.py  contrasts.py
│   ├── report/    build_report.py  viz.py
│   └── cli.py
└── outputs/
```

```
numpy scipy pandas scikit-learn networkx rapidfuzz pyyaml
sentence-transformers  # paraphrase-multilingual-MiniLM-L12-v2 (한국어 정책 필수)
```

임베딩 백엔드 로드 실패 시 TF-IDF로 폴백하되 **사용한 백엔드를 리포트 메타에 기록**한다.
임베딩 임계값은 §4.3의 재보정 절차를 반드시 거친다.

---

## 2. 데이터 계약

### 2.1 단일 pathway 시나리오

```jsonc
{
  "scenario_id": "...",
  "participants": [ {"persona_id","stakeholder_type","name","professional_persona",
                     "skills_and_expertise","career_goals_and_ambitions",
                     "institution":"", "role":"", "background":"",   // 항상 빈 문자열
                     "sex","age","education_level","occupation", ...} ],   // 5명
  "forward_pass": [
    {"phase","direction":"forward","posting_order":[...],       // 단계마다 로테이션
     "initial_posts": [{"persona_name","stakeholder_type",      // persona_id 없음
                        "prediction_values":{...},"narrative","evidence":[...],
                        "judgment": null}],
     "revised_posts": [ /* initial과 전량 동일 */ ],
     "refined_posts": null, "phase_summary": "..."} ],
  "backward_pass": null, "fwd_bwd_fwd_pass": null, "fwd_bwd_fwd_bwd_pass": null,
  "cross_checks": [{"phase","variable","forward_agg","backward_agg":null,
                    "gap":null,"consistency":null,"aggregated_value"}],
  "scenario_impact": {...}, "scenario_confidence": {"<var>": null}
}
```

### 2.2 분기 pathway 트리 (NSA / K-패스)

```jsonc
{ "format_version": 1,
  "generation_mode": "phase_local_independent_children",
  "phase_interaction_mode": "independent",
  "source_run_dir": "...", "scenario_id": "S1_...", "simulation_target": "...",
  "branch_inputs": false,                         // Inputs 미분기 → root 1개
  "phases": ["Inputs","Activities","Outputs","Outcomes","Impact"],
  "transitions": ["enabling","baseline","constraining"],
  "completed_parent_ids": [ ... ],                // 40개 = 1+3+9+27
  "nodes": {
    "root": {"node_id","parent_id":null,"transition_mode":null,"phase_index":0,
             "parent_context_hash","phase_result": {...}},
    "root/baseline/enabling/...": { ... }
  } }
```

**핵심 구조 사실 (NSA 실측)**

| 항목 | 값 |
|---|---|
| 노드 수 | 121 = 1 + 3 + 9 + 27 + 81 |
| 노드 키 | 경로 문자열. `key.count("/") == phase_index` |
| root→leaf 경로 | 81개 |
| Outcomes 노드 | 27개 ← **정확도 평가 단위** |
| 형제 삼중쌍 | 40세트 (= `completed_parent_ids`) ← **B1 검정 단위** |
| `phase_result` | 단일 pathway `forward_pass` 원소와 **동일 스키마** |
| `posting_order` | **121노드 전부 고정** (단일은 로테이션) |
| 페르소나 | 트리 전체가 동일 5인 조합 1개 |
| `revised_posts` | initial과 전량 동일 (no-op) |
| `judgment` / `refined_posts` | 전량 null |

K-패스 트리도 같은 스키마인지 **로더에서 검증**한다(§8.1).

### 2.3 KG

```jsonc
{ "ontology": {"entity_types":[{"name","iad_category","is_actor","persona_eligible",
                               "attributes":[...]}], "edge_types":[...]},
  "nodes": [{"id":"Type:Name","name","type","summary","attributes":{...},
             "source_spans":[{"chunk_id","text"}]}],
  "edges": [{"id","type","edge_category":"structural|regulative|constitutive",
             "source","target","fact","deontic","activation_condition","source_text"}],
  "metadata": {...}, "validation_warnings": [...] }
```

### 2.4 공통 중간표현

```python
@dataclass(frozen=True)
class Post:
    cell_id: str
    run_id: str                       # 단일: scenario_id / 분기: leaf 경로 또는 노드 경로
    path_key: str                     # 단일: "single" / 분기: "root/enabling/..."
    phase: str; phase_index: int
    order_index: int
    agent_key: str                    # persona_name
    stakeholder_type: str
    role_slot: str                    # "AIResearcher#1" (동일 타입 중복 대응)
    prediction_values: dict[str, float]
    narrative: str; evidence: list[str]
    transition_mode: str | None        # 분기 전용

@dataclass(frozen=True)
class Run:
    cell_id: str; run_id: str
    source: Literal["single", "branch"]
    transition_path: tuple[str, ...]   # 단일: ()
    phases: list[PhaseState]
    target_values: dict[str, float]    # GT 단계(Outcomes)의 값
```

---

## 3. 전처리

### 3.1 정책 문서

| 항목 | 100E 실측 | 대응 |
|---|---|---|
| 페이지 마커 `--- page N ---` | 68개 | 페이지 번호 메타 보존 후 제거 |
| 이미지 플레이스홀더 | 19개 | 제거 |
| 다중 공백 (PDF 추출) | 224개 라인 | 축약 + **오프셋 맵 필수** |
| 곡선 아포스트로피 | 51개 | ASCII 정규화 |
| 목차 잔여 숫자 | 7개 라인 | **목차 구간 통째로 제외** |
| 페이지 푸터 | 3개 | 제거 |

BK21(한국어) 추가 확인: 한글 스케일어(천/만/억/조), 각주 번호가 수치에 붙는 경우
(`2,700명1)`), 표에서 추출된 수치의 헤더 소실.

**오프셋 맵**: 정규화 후에도 원문 좌표로 하이라이트해야 한다.

```python
clean_text, offset_map = normalize_with_map(raw_text)
def to_raw(span): return (offset_map[span[0]], offset_map[span[1]-1] + 1)
```

### 3.2 KG

| 항목 | 실측 | 대응 |
|---|---|---|
| `summary` 근중복 문장 | 100E 11/218 노드 | `SequenceMatcher > 0.75` 중복 제거 |
| 중복 엣지 | 16건 | 병합 |
| `source_spans.text`가 패러프레이즈 | 문자열 일치 100E 56%, **BK21 0%** | 원문 인용으로 쓰지 않음 |
| `chunk_id` ↔ 문서 정렬 | metadata 88 vs 실제 71 | fuzzy 정렬 + confidence 기록 |
| 추론 기반 노드 | `validation_warnings` 1,366건 | tier 부여 |

**신뢰도 tier**

| tier | 조건 | 사실 대조(A1) | 근거 풀(A2) |
|---|---|---|---|
| T1 | 원문 문자열 일치 + `evidence_alignment: supported` | ✅ | ✅ |
| T2 | 의미적 정렬 (임베딩 유사도 ≥ 재보정 임계값) | ❌ | ✅ |
| T3 | 그 외 / "inferred from ontology" | ❌ | 별도 채널 |

> **BK21은 T1이 원천적으로 0**이다. KG가 영어, 문서가 한국어라 문자열 일치가 불가능하다.
> 다국어 임베딩으로 개선되는 것은 T2이지 T1이 아니다. 이 한계는 리포트에 명시한다.

### 3.3 시나리오

| 항목 | 대응 |
|---|---|
| `institution`/`role`/`background` 공백 | 페르소나 특성은 자유텍스트 필드에서만 |
| post에 `persona_id` 없음 | `persona_name`으로 조인, 참가자 목록과 대조 검증 |
| 이름 퇴화 (`An`, `A (9s3qtv)`) | **이름 문자열 매칭 인용 탐지 금지** |
| 동일 `stakeholder_type` 중복 | `role_slot = f"{type}#{순번}"` |
| `revised_posts` no-op | `initial_posts`만 사용, assert 후 메타 기록 |
| `judgment`/`refined_posts`/`backward_pass` null | 관련 지표 계산 금지 |
| `aggregated_value` | 5인 산술평균 재계산 검증 |

### 3.4 변수 유형 분류

각 `(phase, variable)`을 분류한다.

| 유형 | 판정 | Accuracy | Diversity |
|---|---|---|---|
| `anchored_constant` | 문서/KG(T1)에 대응 수치 존재 | 값 대조 ✅ | **제외** |
| `predicted` | 문서 근거 없는 미래 추정 | GT 있을 때만 | **계산 대상** |
| `unanchored_constant` | 근거 없는데 분산 0 | 대조 대상 아님 | 제외 |
| `headcount_mislabeled` | 변수명은 rate인데 값이 명·건수 | **정확도 채점 보류** | 서술 통계만 |

> **마지막 유형은 BK21에서 실제로 발견되었다.**
> `graduate_employment_rate_of_bk21_participants_in_2022`의 값 83%가 [0,100]을 벗어났고,
> 확인 결과 실제로 취업자 수(명)를 담고 있었다. 실제 비율은 동일 단계의
> `employment_rate`에 있었다. **이런 변수는 GT 대조와 분산분해 양쪽 모두에서 제외한다.**
> (제외하지 않으면 `D1` 원분산이 1e+20 규모로 오염된다)

### 3.5 수치 정규화

```python
@dataclass(frozen=True)
class Quantity:
    value: float
    kind: str        # currency|count|percent|percentage_point|rate|rank|hours|years
    currency: str | None
    bound: str       # exact | at_most | at_least | range
    range_hi: float | None
    approx: bool
    raw: str
    span: tuple[int, int]
```

**경계 표현 판정** — 정책 문서의 수치는 대부분 점값이 아니다.

| 문서 표현 | bound | 판정 |
|---|---|---|
| `up to S$330,000 per project` | at_most | 발화값 ≤ 330000 |
| `more than S$500 million` | at_least | 발화값 ≥ 5e8 |
| `typically 6-12 months` | range | 구간 포함 |
| `S$330,000` | exact | 상대오차 ≤ 0.5% |

```python
def satisfies(o: Quantity, r: Quantity, rel_tol=0.005) -> bool:
    if o.kind != r.kind: return False
    if o.kind == "currency" and o.currency != r.currency: return False
    tol = rel_tol * max(abs(o.value), abs(r.value), 1.0)
    if o.approx or r.approx: tol *= 4
    match r.bound:
        case "exact":    return abs(o.value - r.value) <= tol
        case "at_most":  return o.value <= r.value + tol
        case "at_least": return o.value >= r.value - tol
        case "range":    return r.value - tol <= o.value <= r.range_hi + tol
```

- 환율 변환 금지 (통화가 다르면 비교 불가)
- `%`와 `%p`는 별개 단위
- 한글 스케일어 필수: `2.6억원 == 260,000,000원`

---

## 4. 참조 인덱스

### 4.1 문서 인덱스

clean_text를 문단 단위로 청킹. KG의 `chunk_id`에 의존하지 않고 독립 구축한다.

### 4.2 KG 인덱스

```python
class KGIndex:
    fact_table: dict[str, list[Fact]]   # T1 노드의 수치 속성
    text_pool:  list[TextRef]           # (source_id, tier, text) — tier별 분리
    role_types: set[str]                # persona_eligible == True
```

### 4.3 임베딩 임계값 재보정 (필수)

신경망 임베딩은 이방성(anisotropy) 때문에 **완전히 무관한 문장 쌍도 코사인 유사도가 높다.**
TF-IDF 임계값(0.35)을 그대로 쓰면 근거 보유율이 0.98까지 치솟는다.

```
1. null 분포 생성: 정책 A의 발화 × 정책 B의 KG, 그리고 그 역방향
   (100E 실측: 중앙값 0.43, p95 0.557)
2. 임계값 = null 분포의 p95  (재보정 결과 0.55)
3. prior 풀 비교도 반드시 같은 임베딩 백엔드로 통일한다.
   TF-IDF와 임베딩을 같은 argmax에서 섞으면 스케일이 달라 비교가 무의미하다.
```

재보정 후에도 "일반적 자기소개 문장"과 "특정 사실 인용 문장"이 0.55~0.7 구간에서
겹치는 한계는 남는다. 소형 다국어 모델의 해상도 한계로 리포트에 명시한다.

---

## 5. 지표 명세

### A1. 문서 모순 (Accuracy · micro)

**목표**: 문서에 없는 항목이 발화에 등장하는 것은 문제가 아니다.
**문서가 명시한 값과 정면으로 어긋나는 발화만** 잡는다.

**탐지 방향이 결정적이다.** 발화에서 출발하면 별칭 사전에 없는 표현을 쓴 모순을 놓친다.
사실에서 출발해야 재현율이 확보된다.

```python
for fact in fact_table:                     # 정책당 20~40개
    for post in all_posts:
        for m in find_alias_mentions(post.text, fact.aliases):
            mention_count[fact.id] += 1
            qs = [q for q in quantities_near(post.text, m.span, window=60)
                  if q.kind == fact.kind]           # 단위 종류 필터 (필수)
            if not qs:
                unscored[fact.id] += 1; continue
            for q in qs:
                verdict = "CONSISTENT" if satisfies(q, fact) else "CONTRADICTION"
                log_evidence(fact, post, q, verdict)   # span·원문 반드시 저장
```

$$
A1 = 1 - \frac{N_{\text{contradiction}}}{N_{\text{consistent}} + N_{\text{contradiction}}}
$$

**함정**: 단위 종류 필터를 빼면 "예산" 별칭 근처의 연도(2023)나 과제 수(125)가
금액과 대조되어 전부 모순으로 찍힌다.

**필수 부속 산출물**: 사실별 진단표(§6.1)와 수기 검토 정밀도(§6.3).
**정밀도 없이 A1 점수만 보고하지 않는다.**

---

### A2. 근거 보유율 (Accuracy · micro)

발화의 각 근거 문장을 4개 풀 중 argmax로 배정한다.

| 출처 | 풀 |
|---|---|
| `doc` | 문서 청크 |
| `kg` | KG 텍스트 풀 (T1·T2) |
| `prior` | 같은 단계 선행 발화 + 직전 단계 `phase_summary` |
| `none` | 최대 유사도 < 재보정 임계값 |

$$
A2 = 1 - \frac{1}{|Z|}\sum_{z \in Z}\text{none\_share}(z)
$$

단계별 3-vector `(doc, kg, prior)`는 서술 통계로 함께 보고한다(점수 아님).
**ToC 구조상 후반 단계가 `prior`에 의존하는 것은 정상이므로 감점하지 않는다.**

---

### M1. 수렴도 (Accuracy · meso)

계산 대상은 `predicted` 변수만.

$$
CV(z,v) = \frac{\sigma_{z,v}}{\max(|\mu_{z,v}|,\ \epsilon)}
\qquad
M1 = 1 - \text{clip}\!\left(\frac{\text{median}_{z,v}\,CV(z,v)}{CV_{\text{ref}}},\ 0,\ 1\right)
$$

$CV_{\text{ref}} = 0.5$. 추세 $\rho_{\text{Spearman}}(z, \text{Convergence}(z))$도 함께.

**집계 방식을 하나로 고정한다.** run별로 계산 후 평균(macro-average)을 쓰고,
셀 요약과 셀 간 대비에서 같은 값이 나오도록 한다.
(v3 리포트에서 §4는 0.978, §5는 0.844로 갈렸던 원인)

---

### M2. 값 인용률 (Accuracy · meso)

이름 매칭 대신 **수치 반향**으로 인용을 탐지한다.

$$
M2 = \frac{1}{n-1}\sum_{k=2}^{n}\mathbb{1}\Big[(\mathcal{M}_k \cap \mathcal{P}_k)\setminus\mathcal{O}_k \ne \emptyset\Big]
$$

$\mathcal{M}_k$ = $k$번째 발화 본문의 수치, $\mathcal{P}_k$ = 선행자 예측값,
$\mathcal{O}_k$ = 자기 예측값. 즉 "내 값이 아니면서 앞사람 값인 숫자를 본문에 언급했는가".

---

### X1. 예측 오차 (Accuracy · macro)

$$
\text{APE} = \frac{|\hat{y} - y_{\text{true}}|}{|y_{\text{true}}|}\times 100
$$

run별 부트스트랩 500회(시드 고정)로 `APE [95% CI]` 보고.

**개념 중복 변수의 상류 채점을 이 표에 행으로 포함한다.** 별도 지표로 두지 않는다.
단, 그룹 멤버가 `headcount_mislabeled`이면 제외한다.

---

### X4. 캘리브레이션 (Accuracy · macro)

```python
coverage_50, coverage_90        # 경험 분위구간이 참값을 포함하는가
PIT  = mean(y_i <= y_true)      # 잘 맞으면 0.5 근처, 0/1이면 체계적 편향
CRPS = mean_i|y_i - y_true| - 0.5 * mean_{i,j}|y_i - y_j|
```

CRPS 앞항은 정확성, 뒷항은 예리함에 대한 보상. 분포를 넓히면 뒷항이 커져 손해다.
**점추정이 같아도 분포 품질이 다른 경우를 잡아내는 것이 이 지표의 존재 이유다.**

---

### D1. 역할 설명 분산 (Diversity · micro)

**v3에서 재정의됨**: 원분산이 아니라 급내상관계수를 쓴다.

$$
\text{Var}_{\text{total}} = \text{Var}_{\text{within-role}} + \text{Var}_{\text{between-role}} + \text{Var}_{\text{between-run}}
$$

$$
D1 = \text{ICC}_{\text{role}} = \frac{\text{SS}_{\text{between-role}}}{\text{SS}_{\text{total}}},
\qquad
\text{SS}_{\text{between-role}} = \sum_r n_r(\bar{y}_r - \bar{y})^2
```

변수별로 z-score 표준화 후 계산한다. 표준화하지 않으면 스케일 큰 변수가 지배한다.
`headcount_mislabeled` 변수는 제외한다.

`ICC_run`도 함께 보고한다.

---

### D2. 의미 분산 (Diversity · micro)

$$
D2 = \frac{2}{n(n-1)}\sum_{i<j}\big(1 - \cos(\mathbf{v}_i, \mathbf{v}_j)\big)
$$

TF-IDF 벡터 기준. 임베딩 백엔드가 있으면 병행하되 **TF-IDF 값을 항상 함께 보고**한다.

---

### DM1. 상호작용 구조 다양성 (Diversity · meso)

run × 단계마다 영향 그래프를 만든다.

$$
w_{ij} = 0.5\cdot\mathbb{1}[\text{값 인용}] + 0.2\cdot\mathbb{1}[\text{이름 인용}] + 0.3\cdot\text{shrinkage}_{ij}
$$

$$
\text{shrinkage}_{ij} = \frac{\max\big(0,\ |\hat{y}_j^{\text{base}} - y_i| - |y_j - y_i|\big)}{\text{scale}}
$$

4개 위상 유형으로 분류.

| 유형 | 규칙 |
|---|---|
| `disconnected` | 총 엣지 가중치 < 0.15 |
| `star` | 최대 in-degree/평균 ≥ 2.0, 앵커 1명이 유입 50%↑ |
| `chain` | 엣지 70%↑가 직전 발화자에서 |
| `complete` | 위 해당 없고 밀도 ≥ 0.5 |

$$
DM1 = \frac{H(\text{유형 분포})}{\log 4}
$$

앵커 역할의 `stakeholder_type` 분포는 서술 통계로 함께 보고한다(점수 아님).

---

### DX1. 결과 공간 폭 (Diversity · macro)

$$
DX1 = \frac{\sigma\big(\{\hat{y}^{(r)}\}_{r=1}^{R}\big)}{\max(|\bar{\hat{y}}|,\ \epsilon)}
$$

Outcomes 단계 타깃 변수의 run 간 변동계수. range와 IQR도 함께 보고.

**X4와 짝지어 해석한다.** DX1이 작은데 coverage가 False이고 PIT가 0 또는 1이면
"결과 공간을 좁게 탐색해서 참값을 놓쳤다"는 인과 서사가 성립한다.
**다양성 부족이 정확도 실패의 원인임을 보이는 유일한 연결고리다.**

---

### 게이트 G1 · G2 · G3

점수가 아니라 통과/실패로 한 줄 보고한다.

**G1 체인 일관성**

```yaml
chain_constraints:
  - "Outputs.ai_prototypes_developed   <= Outputs.approved_100e_projects"
  - "Outputs.enterprises_with_solution <= Outputs.approved_100e_projects"
  - "Outcomes.deployed_100e_projects   <= Outputs.approved_100e_projects"
concept_duplicates:      # 같은 개념 변수 간 값 일치 (headcount 변수는 제외)
  - {group: approved_projects_2023,
     members: ["Outputs.approved_100e_projects",
               "Outcomes.number_of_100e_projects_approved_in_2023"]}
```

`ast.parse`로 안전 평가. `eval` 금지.

**G2 값 범위 타당성** — 비율 [0,100], 순위 ≥ 1 등. 위반 변수명을 함께 출력한다.
BK21의 `headcount_mislabeled` 발견이 이 게이트에서 나왔다.

**G3 산출 완료율** — 파싱 실패·null 비율.

보고 형식:

> 데이터 타당성 게이트: 체인 일관성 통과(위반 6/150) · 값 범위 통과(0/1250) · 완료율 100%

---

## 6. A1 수기 검토 프로토콜 ★

A1은 파이프라인에서 **유일하게 사람 손이 필요한 지표**다. 총 1시간 정도 소요된다.
이 절차를 거치지 않은 A1 점수는 보고하지 않는다.

### 6.1 Step 1 — 사실별 진단표 (코드, 5분)

`src/audit/a1_diagnostic.py`가 사실 테이블 항목별로 아래 표를 출력한다.

| fact_id | mention | with_qty | consistent | contradiction | unscored |
|---|---|---|---|---|---|
| co_funding_cap | 210 | 195 | 180 | 15 | 15 |
| total_rd_commitment | 340 | 320 | 54 | 266 | 20 |

**용어 정의**

- `mention` — 그 사실의 별칭이 등장한 발화 수. **판정의 모수**
- `with_qty` — 별칭 근처(±60자)에 같은 단위 종류의 수치가 있던 경우
- `unscored` — 별칭은 잡혔으나 비교할 수치가 없어 판정 불가

**진단 규칙**

| 패턴 | 원인 | 대응 |
|---|---|---|
| `mention` ≈ 0 | 별칭이 발화 표현과 불일치 | Step 2에서 별칭 보강 |
| `mention` 큼, `with_qty` 작음 | 언급은 하나 수치를 안 붙임 / 단위 필터 | 별칭 보강으로 안 고쳐짐 |
| `contradiction` 비율 이상 높음 | 스코프 파생 또는 매칭 오류 의심 | Step 3에서 집중 표본 |

> BK21의 `unscored = 348 / 613`(57%)이 어느 패턴인지 **먼저 확인**한다.
> 원인을 모르는 상태에서 별칭을 늘리거나 문서를 번역하면 헛수고할 수 있다.

### 6.2 Step 2 — 별칭 보강 (수기, 20분)

`mention`이 0이거나 한 자릿수인 사실만 골라, 해당 사실을 다루는 발화 몇 개를 열어
**에이전트가 실제로 쓰는 표현**을 확인하고 별칭에 추가한다.

BK21은 문서가 한국어, 발화가 영어이므로 **영어 별칭을 병기**해야 매칭된다.
(한국어 별칭만 있었을 때 A1 매칭 0건)

**별칭은 넉넉하게 넣는다.** 모순 탐지가 목적이므로 미탐이 오탐보다 훨씬 나쁘다.
오탐은 근거 span이 저장되어 있어 Step 3에서 걸러낼 수 있지만, 미탐은 존재를 알 수 없다.

### 6.3 Step 3 — 모순 표본 라벨링 (수기, 30~40분) ★ 핵심

`src/audit/a1_sampler.py`가 `contradiction` 판정 중 **무작위 100건**을 CSV로 내린다.

| 컬럼 | 내용 |
|---|---|
| `idx` | 일련번호 |
| `fact_id` | 사실 식별자 |
| `cell_id` / `run_id` / `phase` / `agent` | 출처 |
| `utterance` | 발화 원문 (수치 span 앞뒤 200자) |
| `doc_value` / `doc_bound` | 문서 기준값과 경계 종류 |
| `uttered_value` | 발화에서 추출된 값 |
| `label` | **사람이 채우는 칸 (A/B/C)** |

**라벨 정의**

| 라벨 | 의미 | 예시 |
|---|---|---|
| **A** | 진짜 모순 | 문서 "up to S$330,000" ↔ 발화 "capped at S$500,000" |
| **B** | 스코프 파생 | 문서 "more than S$500M" ↔ 발화 "national commitment exceeds S$500M, but 2023 R&D allocation is ~S$180M" — 문서값을 인지한 채 하위 범위 추정 |
| **C** | 매칭 오류 | "예산" 별칭 근처의 과제 수 125를 금액과 대조 |

$$
\text{Precision} = \frac{|A|}{100}
$$

**보고 형식**

> A1 = 0.532 (모순 판정 100건 수기 검토: 진짜 모순 N건, 스코프 파생 M건, 매칭 오류 K건)

이 한 줄이 있으면 "그 모순이 진짜인가"라는 질문이 봉쇄된다.
없으면 A1 점수 전체를 쓸 수 없다.

### 6.4 Step 4 — 배제 규칙 추가 (선택, 15분)

B 라벨 사례에서 공통 신호어를 뽑아 배제 필터를 만든다.

```yaml
scope_derivation_markers:
  - allocation | allocated for | portion of | share of
  - estimate | estimated | projected
  - "in 2023" | "for this year" | annual
  - per institution | per university | 기관별 | 연차별
```

같은 문장에 이 표현이 있으면 판정을 보류(`SCOPE_DERIVED`)하고 분모에서 뺀다.
적용 후 재실행하면 A1이 상승한다.

**시간이 없으면 Step 4는 건너뛴다.** 한계를 밝히고 정밀도를 보고하는 편이
억지로 점수를 올리는 것보다 정직하고 인상도 낫다.

---

## 7. 실험 설계 (단일 pathway)

```yaml
# configs/cells.yaml
cells:
  - {id: 100e_gemini,   policy: 100e, policy_type: distributive, model: gemini,   n: 50}
  - {id: 100e_deepseek, policy: 100e, policy_type: distributive, model: deepseek, n: 50}
  - {id: bk21_deepseek, policy: bk21, policy_type: distributive, model: deepseek, n: 50}
```

**불완전 2×2 설계다.**

| 대비 | 방법 | 가능? |
|---|---|---|
| 모델 효과 | `100e_gemini` vs `100e_deepseek` (정책 고정) | ✅ |
| 정책 효과 | `100e_deepseek` vs `bk21_deepseek` (모델 고정) | △ **교란 있음** |
| 상호작용 | 정책 × 모델 | ❌ `bk21_gemini` 부재 |

> **정책 효과 대비는 KG 품질 교란을 반드시 각주로 단다.**
> BK21은 KG 노드가 100E의 1/4(49 vs 218)이고 T1이 0개다.
> A1 차이를 "정책 유형 효과"로 해석하면 안 된다.

**집계 규칙**

```
1. 셀을 가로질러 원자료를 풀링하지 않는다 (모델 효과가 크다)
2. 주 결과표는 셀별 행
3. 모든 지표에 부트스트랩 500회 CI (시드 고정)
4. 셀 요약과 셀 간 대비가 같은 집계 방식(run별 평균)을 쓰도록 통일
5. 세 셀 모두 n=50 균형 → 엔트로피 계열 rarefaction 불필요
```

**안정성 곡선**: n=50은 안정화 임계점에 걸쳐 있다.
셀마다 N ∈ {10,20,30,40,50} 서브샘플 학습곡선(부트스트랩 500회, ±1σ)을 그린다.

---

## 8. 분기 pathway 확장 ★

NSA·K-패스 트리를 동일 파이프라인에 태우기 위한 명세.

### 8.1 로더 (`src/branch/tree_loader.py`)

```python
def load_branch_tree(path: str, cell_id: str) -> BranchTree:
    d = json.load(open(path))

    # --- 스키마 검증 (K-패스가 NSA와 같은 구조인지 확인) ---
    assert d["format_version"] == 1
    assert d["phases"] == ["Inputs","Activities","Outputs","Outcomes","Impact"]
    assert set(d["transitions"]) == {"enabling","baseline","constraining"}
    assert d["branch_inputs"] is False
    n = d["nodes"]
    assert len(n) == 121, f"노드 수 이상: {len(n)}"
    for k, v in n.items():
        assert k.count("/") == v["phase_index"]

    # --- no-op / null 검증 (단일과 동일) ---
    for v in n.values():
        pr = v["phase_result"]
        assert pr["initial_posts"] == pr["revised_posts"]      # revise no-op
        assert pr["refined_posts"] is None
        assert all(p["judgment"] is None for p in pr["initial_posts"])

    # --- 중복 발화 탐지 (NSA 실측: 605개 중 고유 582개) ---
    dup = detect_duplicate_narratives(n)                        # 메타에 기록

    return BranchTree(nodes=n, meta=d, duplicate_narratives=dup)
```

**K-패스에서 스키마가 다르면** 어떤 필드가 다른지 로그로 남기고 중단한다.
조용히 넘어가면 이후 지표가 전부 오염된다.

### 8.2 경로 추출 (`src/branch/paths.py`)

```python
def extract_runs(tree: BranchTree, level: str) -> list[Run]:
    """
    level='outcomes' : Outcomes 노드 27개 → 정확도(X1, X4)·DX1용
    level='leaf'     : leaf 노드 81개    → 다양성(D2, DM1)용
    각 노드에서 root까지 부모를 거슬러 phase_result를 모아 Run을 만든다.
    """
    targets = [k for k in tree.nodes if k.count("/") == LEVEL_DEPTH[level]]
    runs = []
    for key in targets:
        chain, cur = [], key
        while cur is not None:
            chain.append(tree.nodes[cur])
            cur = tree.nodes[cur]["parent_id"]
        chain.reverse()
        runs.append(build_run(chain, transition_path=tuple(key.split("/")[1:])))
    return runs
```

**표본 단위가 지표마다 다르다는 점을 리포트에 명시한다.**

| 지표 | 표본 단위 | n |
|---|---|---|
| X1, X4, DX1 | Outcomes 노드 | 27 (n_eff 보정) |
| A1, A2, M1, M2, D1, D2, DM1 | 전체 노드의 발화 | 121 노드 × 5 |
| B1 | 형제 삼중쌍 | 40 |

### 8.3 상관 보정 (필수)

분기 Run은 조상 단계를 공유하므로 독립 표본이 아니다.

$$
\text{ICC}_{\text{subtree}} = \frac{\text{Var}_{\text{between-subtree}}}{\text{Var}_{\text{between-subtree}} + \text{Var}_{\text{within-subtree}}}
\qquad
n_{\text{eff}} = \frac{n}{1 + (m-1)\cdot \text{ICC}_{\text{subtree}}}
$$

$m$ = 평균 하위트리 크기 (Outcomes 27개 기준 depth-1 하위트리당 9).
**모든 분기 통계에 $n_{\text{eff}}$를 병기한다.**

### 8.4 지표 적용 가능성

| 지표 | 분기 적용 | 비고 |
|---|---|---|
| A1 문서 모순 | ✅ | 그대로 |
| A2 근거 보유율 | ✅ | `prior` 풀에 부모 노드의 `phase_summary` 포함 |
| M1 수렴도 | ✅ | |
| M2 값 인용률 | ✅ | |
| X1 예측 오차 | △ | Outcomes 노드 27개, $n_{\text{eff}}$ 병기 |
| X4 캘리브레이션 | △ | 동일 |
| D1 역할 설명 분산 | ✅ | 121노드가 반복 관측 역할. 페르소나 조합은 1개 |
| D2 의미 분산 | ✅ | 중복 발화 가중치 축소 적용 |
| DM1 구조 다양성 | △ | **`posting_order`가 121노드 고정** → `chain` 유형 과대 추정.<br>단일과 직접 비교 금지, `chain` 비율 별도 표시 |
| DX1 결과 공간 폭 | ✅ | 분산 출처가 다름(가정 조건부 vs 페르소나 샘플링) — 해석 시 명시 |
| G1·G2·G3 | ✅ | |
| **B1 라벨 충실도** | **분기 전용** | §8.5 |

**단일 pathway와의 직접 비교 시 주의**

| | 단일 | 분기 |
|---|---|---|
| 분산 출처 | 페르소나 샘플링 | 가정 조건부 분기 |
| 페르소나 조합 | 50개 | **1개** |
| 발화 순서 | 단계마다 로테이션 | **전부 고정** |

D1과 DX1은 양쪽에서 계산되지만 **재는 대상이 다르다.** 같은 표에 나란히 놓지 말고
"어떤 분산을 재는가"를 열로 구분한다.

### 8.5 B1 — 분기 라벨 충실도 (`src/branch/label_fidelity.py`)

**무엇을 재는가**: `enabling` / `baseline` / `constraining` 가정 라벨이
실제 결과와 부호가 맞는가.

$$
B1 = 1 - \frac{|\{p : \neg(y^{\text{enab}}_p \succeq y^{\text{base}}_p \succeq y^{\text{cons}}_p)\}|}{|\{p\}|}
$$

$p$ = 형제 삼중쌍, $\succeq$ = 지표 방향 보정 후 비교.

**통계 검정**: 형제 삼중쌍 40세트는 **짝지어진 설계**다. 트리가 1개여도 검정이 성립한다.

$$
\chi^2_F = \frac{12n}{k(k+1)}\sum_{j=1}^{k}\bar{R}_j^2 - 3n(k+1) \qquad (k=3)
$$

Friedman 검정 후 사후 비교는 Wilcoxon 부호순위 + 본페로니 보정.

```python
def label_fidelity(tree, target_var, direction="higher_is_better"):
    triples = []
    for parent in tree.meta["completed_parent_ids"]:
        vals = {}
        for t in ["enabling", "baseline", "constraining"]:
            key = f"{parent}/{t}" if parent != "root" else f"root/{t}"
            vals[t] = aggregate(tree.nodes[key], target_var)   # 5인 평균
        triples.append(vals)
    violations = sum(1 for v in triples if not monotone(v, direction))
    stat, p = scipy.stats.friedmanchisquare(
        *[[t[m] for t in triples] for m in ["enabling","baseline","constraining"]])
    return {"B1": 1 - violations/len(triples), "friedman_chi2": stat, "p": p,
            "n_triples": len(triples), "medians": {...}}
```

> ⚠️ **NSA 실측 이상**: 응급 OON 감소폭 중앙값이
> enabling 8.0 / baseline 8.5 / **constraining 11.5**로 라벨과 부호가 반대다.
> 원인은 ① 지표 방향 정의 혼선(감소폭↑ = 개선) 또는
> ② 프롬프트에서 `constraining`이 "정책 성공에 불리한"이 아니라
> "시스템 부하가 큰"으로 해석되었을 가능성.
> **B1 구현 전에 `direction` 설정을 확정한다.** 이게 어긋나면 분기 지표 전체가 해석 불가다.

### 8.6 분기 실험 셀

```yaml
branch_cells:
  - {id: nsa_deepseek,   policy: nsa,   policy_type: regulatory,   model: deepseek, trees: 1}
  - {id: kpass_deepseek, policy: kpass, policy_type: distributive, model: deepseek, trees: 1}
```

**두 셀은 경로 구조와 모델이 통제된 비교다.** 따라서 여기서 나오는 차이는
설계 탓이 아니라 정책 유형 탓으로 귀속할 수 있다.
정책 유형별 해석기준(별도 문서)의 예비 검증에 사용한다.

주 비교 지표: M1(수렴도), D1(역할 설명 분산), DM1(구조 다양성).
분배형(K-패스)은 수렴도가 높고, 규제형(NSA)은 낮을 것으로 사전 예측한다.

---

## 9. 출력 스키마

```jsonc
{
  "meta": {"cells": [...], "branch_cells": [...],
           "eval_terminal_phase": "Outcomes", "post_target_phases": ["Impact"],
           "retriever_backend": "embedding", "embedding_threshold": 0.55,
           "threshold_calibration": {"null_median": 0.43, "null_p95": 0.557},
           "seed": 20260810,
           "revise_noop_verified": true, "aggregated_value_matches_mean": true,
           "kg_tier_counts": {"100e": {...}, "bk21": {...}},
           "variable_typing": {"anchored": .., "predicted": ..,
                               "unanchored": .., "headcount_mislabeled": [...]}},

  "by_cell": {
    "<cell_id>": {
      "n_runs": 50,
      "A1": {"score": .., "consistent": .., "contradiction": ..,
             "unscored": .., "by_fact": {...},
             "audit": {"sampled": 100, "true_contradiction": ..,
                       "scope_derived": .., "match_error": .., "precision": ..}},
      "A2": {"score": .., "by_phase": {"Inputs": {"doc":..,"kg":..,"prior":..,"none":..}}},
      "M1": {"score": .., "trend": ..},
      "M2": {"score": ..},
      "X1": {"<target>": {"APE_mean": .., "APE_median": .., "CI95": [.., ..],
                          "source": "direct|concept_duplicate"}},
      "X4": {"<target>": {"coverage_50": .., "coverage_90": .., "PIT": .., "CRPS": ..}},
      "D1": {"ICC_role": .., "ICC_run": ..},
      "D2": {"score": ..},
      "DM1": {"score": .., "topology": {...}, "anchor_role_dist": {...}},
      "DX1": {"<target>": {"CV": .., "range": [.., ..], "IQR": ..}},
      "gates": {"G1": {"pass": true, "violations": 6, "total": 150},
                "G2": {"pass": true, "violations": 0, "total": 1250,
                       "violating_vars": []},
                "G3": {"completion_rate": 1.0}},
      "stability_curve": {"10": .., "20": .., "30": .., "40": .., "50": ..}
    }
  },

  "by_branch_cell": {
    "<cell_id>": {
      "n_outcomes_nodes": 27, "n_eff": .., "n_leaves": 81, "n_triples": 40,
      "duplicate_narratives": .., "posting_order_fixed": true,
      "<지표들 — by_cell과 동일 구조>": ...,
      "B1": {"score": .., "friedman_chi2": .., "p": ..,
             "medians": {"enabling": .., "baseline": .., "constraining": ..},
             "direction": "higher_is_better"}
    }
  },

  "contrasts": {
    "model_effect_100e": {...},
    "policy_effect_deepseek": {"...": ..., "confound_note": "BK21 KG 품질 열위"},
    "branch_policy_type": {"nsa_vs_kpass": {...}},
    "interaction": null
  },

  "evidence": [
    {"id": "ev_0001", "metric": "A1", "verdict": "CONTRADICTION",
     "cell_id": "...", "run_id": "...", "phase": "Inputs",
     "agent_key": "...", "role_slot": "...",
     "utterance_span_raw": [.., ..], "utterance_text": "...",
     "parsed": {"value": .., "kind": "currency", "currency": "SGD"},
     "reference": {"source": "doc", "page": .., "bound": "at_least", "value": ..,
                   "text": "..."},
     "audit_label": "A|B|C|null"}
  ]
}
```

---

## 10. 구현 순서

| # | 산출물 | 비고 |
|---|---|---|
| 1 | `normalize.py` (경계 인지 + 한글 스케일) | 정확도 전체가 의존 |
| 2 | `doc_clean.py` + 오프셋 맵 | 목차 노이즈 제거 |
| 3 | `loaders.py` → IR, no-op assert | |
| 4 | `var_typing.py` (headcount 유형 포함) | 다양성 지표의 전제 |
| 5 | `kg_clean.py` + 임계값 재보정 (§4.3) | |
| 6 | **A1 + `audit/` 진단·표본 도구** | §6 프로토콜 병행 |
| 7 | M1 · M2 | |
| 8 | X1 · X4 · **DX1** | 포스터 핵심 |
| 9 | D1(ICC) · D2 · DM1 | |
| 10 | 게이트 G1·G2·G3 | |
| 11 | `design/contrasts.py` | |
| 12 | **`branch/` 3종 + B1** | 분기 실험 |
| 13 | `viz.py` | |

---

## 11. 함정 체크리스트

**전처리**
- [ ] 목차 구간 미제거 → 수치 추출 오염
- [ ] 오프셋 맵 미보존 → 시각화 하이라이트 어긋남
- [ ] 임베딩 임계값을 TF-IDF 값(0.35) 그대로 사용 → 근거 보유율 0.98로 폭등
- [ ] `prior` 풀만 TF-IDF로 두고 나머지는 임베딩 → 스케일이 다른 argmax 비교
- [ ] KG T3(추론 노드)를 사실 대조에 사용
- [ ] BK21 T1이 0인 것을 버그로 취급 → 언어 불일치로 원천 불가

**A1**
- [ ] 단위 종류 필터 누락 → 연도·개수가 금액과 대조되어 모순 폭증
- [ ] 수기 검토 없이 점수 보고
- [ ] `mention` 진단 없이 별칭 보강 또는 문서 번역
- [ ] 별칭을 좁게 잡아 미탐 발생

**변수·집계**
- [ ] `headcount_mislabeled` 변수를 GT 대조나 분산분해에 포함 → 원분산 1e+20
- [ ] `anchored_constant`를 다양성 계산에 포함
- [ ] 셀 요약과 셀 간 대비가 다른 집계 방식 → 같은 지표가 두 값
- [ ] 셀 가로질러 원자료 풀링

**분기**
- [ ] K-패스 스키마를 검증 없이 NSA와 동일 가정
- [ ] Outcomes 27개를 독립 표본 취급 → $n_{\text{eff}}$ 병기 필수
- [ ] 분기와 단일의 DM1을 직접 비교 → `posting_order` 고정 차이
- [ ] B1의 `direction` 미확정 상태로 해석 → constraining 부호 역전
- [ ] 분기 D1·DX1을 단일과 같은 표에 병치 → 분산 출처가 다름
