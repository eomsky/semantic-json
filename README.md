# Semantic JSON Transport

> **기존 주류 방식인 Fixed Chunk Retrieval이 아닌, query에 따라 원문에서 최적의 EvidenceRegion을 동적으로 구성하는 Retrieval을 지향합니다.**

Semantic JSON Transport는 여신 심사와 같이 정밀도가 높은 상황이 요구될 때 **query-conditioned EvidenceRegion**을 구성하기 위한 RAG 프로젝트로 기획하였습니다.

본 프로젝트에서 고민한 핵심 문제의식은 아래와 같습니다.

> **Vector DB에 저장하는 단위와 최종 LLM context로 전달하는 단위가 반드시 같아야 할까?**

일반적인 RAG는 ingestion 시점에 문서를 fixed chunk로 나눈 뒤 query가 들어오면 Top-k chunk를 검색합니다. 이 프로젝트는 fine-grained `SemanticUnit`을 source-grounded 형태로 보존하고, query가 들어온 뒤 관련 위치를 찾은 다음 원문 위에서 EvidenceRegion을 구성하는 방향을 연구합니다.

## 설치

```bash
pip install semantic-json-transport
```

현재 패키지 버전: **0.2.0a4**


## 빠른 시작

```python
from semantic_json import SemanticRepository

text = """
A기업은 자동차 부품을 생산하고 있으며 최근 해외 매출이 증가하고 있다.
신규 생산라인 가동으로 생산능력도 확대될 전망이다.
다만 원재료 가격 상승으로 수익성에는 일부 부담이 존재한다.

B기업은 산업용 장비를 제조하며 주요 거래처와 장기 공급계약을 체결하고 있다.
최근 매출은 안정적인 수준을 유지하고 있으나 특정 거래처에 대한 의존도가 높다.
해당 주요 거래처와의 공급계약은 내년 말 만료될 예정이다.
계약 갱신 여부는 아직 확정되지 않았다.
최근 영업현금흐름도 전년 대비 감소하였다.
차입금 상환 부담까지 고려하면 중장기 상환능력을 낙관하기 어렵다.

C기업은 소프트웨어 서비스를 제공하고 있으며 구독형 매출 비중이 확대되고 있다.
최근 신규 고객사가 증가하면서 매출 성장세가 이어지고 있다.
현금성자산도 충분하여 단기적인 유동성 위험은 낮은 수준이다.
"""

repo = SemanticRepository()

repo.add_text(
    text,
    document_id="credit_review_sample",
    source_uri="documents/credit_review_sample.txt",
)

result = repo.search(
    "B기업의 중장기 상환능력에 영향을 미치는 요인은?",
    top_k=5,
)

print(result.to_text())
```

가벼운 fallback이 필요하면:

```python
from semantic_json import LiteEmbedder, SemanticRepository

repo = SemanticRepository(
    embedder=LiteEmbedder(),
    region_model="lite",
)
```

## 현재 아키텍처

```text
Offline indexing / hierarchy / reusable embeddings
                    ↓
              Navigation
          어디를 볼 것인가?
                    ↓
       Local EvidenceRegion Composer
          Φθ(Q, R, ∂R)
                    ↓
          source-contiguous basins
             R1*, R2*, ... Rm*
                    ↓
        Global Evidence-Set Value
              Vω(Q, E)
                    +
           Structured Search
                    ↓
             Evidence Set E*
                    ↓
              downstream LLM
```

### 1. Navigation

Navigation은 관련 source location 후보를 찾습니다. Dense retrieval, source adjacency, semantic shortcut, document hierarchy 등을 활용할 수 있습니다.

### 2. Local Composition

Local EvidenceRegion은 원문의 연속 구간입니다.

```text
R = [i, j]
```

현재 연구 formulation은 boundary-aware scalar potential을 학습합니다.

```text
Φθ(Q, R, ∂R)
```

이전 independent edit-utility formulation에서는 cyclic trajectory가 관찰되었습니다. 현재 방식은 neighboring edit를 potential difference로 비교합니다.

```text
ΔQ(R → R') = ΦQ(R') - ΦQ(R)
```

finite Region state space에서 strict potential ascent는 directed cycle을 만들 수 없습니다.

### 3. Global Composition

Local potential `Φ`는 서로 멀리 떨어진 basin을 비교하는 globally calibrated relevance score로 사용하지 않습니다. Global selection은 별도의 set-value 문제로 둡니다.

```text
Vω(Q, E)
```

현재 연구 구현은 **Unary relevance foundation + learned set-interaction residual**과 structured beam search를 결합합니다.

## v0.17 핵심 실험 결과

QASPER 기반 phase-1 benchmark에서는 약한 fixed-window baseline이 아니라 validation에서 충분히 튜닝한 conventional RAG와 비교했습니다.

### Strong Fixed-RAG tuning

v0.17에서는 다음 경우의 수를 validation에서 반복 평가했습니다.

- context budget: `512, 1024, 2048, 4096`
- fixed chunk length: `128, 256, 512, 1024`
- overlap: `0, 0.1, 0.2`
- dense candidate k: `5, 10, 20, 40`
- reranker final k: `1, 3, 5, 8, 10`

`final_k <= dense_k`, `chunk_length <= budget` 조건을 적용했습니다. Dense baseline에는 exact cosine retrieval을 사용했고, 더 강한 baseline에는 query-conditioned neural reranking을 추가했습니다.

### Held-out test

1,309개의 held-out test query에서 관찰한 결과입니다.

| Method | Paragraph Evidence F1 | Internal Unit F1 | Mean actual context tokens |
|---|---:|---:|---:|
| Best-Tuned Dense Fixed RAG | 0.2387 | 0.1757 | 125.9 |
| Best-Tuned Reranked Fixed RAG | 0.2432 | 0.1804 | 127.1 |
| **Query-Conditioned Evidence Construction** | **0.2694** | **0.2274** | 266.7 |

Reranked Fixed RAG 대비 paragraph Evidence F1 차이는:

```text
Δ = +0.0262
95% paired-bootstrap CI = [0.0065, 0.0463]
bootstrap P(Δ <= 0) = 0.0052
```

Internal unit F1 차이는 약 `+0.0470`, 95% CI `[0.0300, 0.0646]`였습니다.

