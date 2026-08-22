# Semantic JSON Transport

> **LLM이 읽기 전에, 의미와 근거 위치를 보존합니다.**  
> **Preserve meaning and source provenance before the LLM sees it.**

Semantic JSON Transport는 장문의 자연어 문서에서 작은 semantic anchor를 검색하고, 질의 시점에 원문 EvidenceRegion을 동적으로 조립한 뒤 구조화된 transport로 전달하는 경량 retrieval layer입니다.

기본 설치는 생성형 LLM, GPU, PyTorch, Transformers 또는 외부 모델 다운로드를 요구하지 않습니다.

## Core principle

> **Index small semantics; transport original evidence.**

Semantic unit은 최종 chunk가 아니라 원문으로 돌아가기 위한 anchor입니다. 최종 EvidenceRegion은 query 이후 조립되며 원문의 정확한 위치를 유지합니다.

## Installation

```bash
pip install semantic-json-transport
```

기본 dependency는 NumPy입니다. 더 강한 neural embedding이 필요하면:

```bash
pip install "semantic-json-transport[transformers]"
```

현재 버전은 `0.1.0a7` alpha입니다.

## Quick Start

```python
from semantic_json import compile, SemanticRepository

text = """
B기업은 현재까지 원리금을 정상적으로 상환하고 있다.
다만 주요 거래계약이 내년에 만료될 예정이며,
중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다.
"""

doc = compile(
    text,
    document_id="company_b",
    source_uri="file:///credit/company_b.txt",
)
repo = SemanticRepository()
repo.add(doc)

result = repo.search("B기업의 중장기 채무상환능력은 어떤가?")

# Canonical structured transport
print(result.to_json())

# Human-readable retrieval inspection / plain-text LLM context
print(result.to_text())
```

`SearchResult`는 structured transport 객체이면서 기존 region list 사용법을 최대한 유지합니다.

```python
first_region = result[0]
for region in result:
    print(region.document_id, region.text)
```

## Canonical structured transport

기본 검색 결과는 versioned schema를 가진 구조화된 객체입니다.

```json
{
  "schema": "semantic-json-transport/context/v1",
  "query": "B기업의 중장기 채무상환능력은 어떤가?",
  "region_count": 2,
  "document_count": 1,
  "documents": [
    {
      "document_id": "doc_002",
      "source_uri": "file:///credit/doc_002.txt",
      "document_sha256": "...",
      "regions": [
        {
          "region_id": "doc_002:R1",
          "score": 0.87,
          "entities": ["B_CORP"],
          "source": {
            "document_id": "doc_002",
            "uri": "file:///credit/doc_002.txt",
            "start_char": 12873,
            "end_char": 13921,
            "start_line": 120,
            "end_line": 145,
            "document_sha256": "..."
          },
          "anchors": ["P14", "P15"],
          "text": "원문 그대로..."
        }
      ]
    }
  ]
}
```

하나의 문서에서 관련 evidence가 여러 곳에 떨어져 있으면 여러 EvidenceRegion으로 유지하고 structured output에서 같은 document 아래 그룹화합니다.

## Plain-text inspection

JSON은 machine/LLM transport의 canonical representation이고, 평문은 retrieval 품질을 사람이 빠르게 확인하는 first-class inspection mode입니다.

```python
print(result.to_text())
# 또는 기존 API
print(repo.build_context(result))
```

평문에는 document, line/character coordinates, score, anchor/entity와 원문 region text가 표시됩니다.

## Source provenance and evidence verification

EvidenceRegion의 원문 좌표가 canonical reference입니다. `text`는 해당 좌표에서 복원된 원문입니다.

```python
region = result[0]

repo.locate(region)
# document_id / source URI / start-end char / line / SHA-256

repo.get_source(region)
# region의 정확한 원문

repo.get_source(region, context_before=500, context_after=500)
# 사람이 근거를 검토할 수 있도록 앞뒤 원문 포함

repo.verify_source(region)
# True: document hash와 exact source slice가 일치
```

이를 통해 downstream LLM의 답변에서 EvidenceRegion을 다시 원문 위치로 연결하는 audit/provenance UI를 구축할 수 있습니다.

## Retrieval architecture

```text
Long Documents
    ↓
Semantic Compiler
    ↓
small source-grounded semantic units
    ↓
semantic index
    ↓
Query
    ↓
semantic anchor retrieval
    ↓
entity-safe source expansion / clustering
    ↓
query-time EvidenceRegion assembly
    ↓
SearchResult
    ├─ structured JSON transport (canonical)
    ├─ plain-text inspection
    └─ source locator / verification
    ↓
downstream LLM / agent / audit UI
```

문법 적용 단위와 최종 retrieval 단위는 동일할 필요가 없습니다. Proposition은 작은 semantic anchor이고 EvidenceRegion은 query 이후 만들어지는 원문 context입니다.

## Retrieval backends

### LiteEmbedder — default

NumPy와 deterministic hashing 및 작은 한·영 semantic normalization lexicon을 사용합니다. 외부 모델 다운로드가 없습니다.

### MultilingualE5Embedder — optional

`sentence-transformers`와 `intfloat/multilingual-e5-small`을 사용하는 선택형 backend입니다.

```python
from semantic_json import SemanticRepository, MultilingualE5Embedder
repo = SemanticRepository(embedder=MultilingualE5Embedder())
```

Embedding backend는 Semantic JSON Transport의 semantic/provenance contract와 분리되어 있습니다.

## Current limitations

현재 alpha는 긴 plain text를 대상으로 합니다. 표, 이미지, PDF layout은 아직 처리하지 않습니다. Rule-based compiler와 LiteEmbedder는 완전한 자연어 이해 시스템이 아닙니다. 특히 범용 entity resolution과 discourse grammar는 향후 강화 대상입니다.

## Roadmap

- [x] Long plain-text → SemanticDocument
- [x] Source-span provenance
- [x] Semantic JSON Grammar v0.1
- [x] NumPy-only LiteEmbedder
- [x] Optional sentence-transformers backend
- [x] Query-time EvidenceRegion assembly
- [x] Entity-safe region expansion
- [x] Canonical structured JSON transport
- [x] Plain-text retrieval inspection
- [x] Source locator / recovery / verification
- [ ] Fixed-chunk vector RAG comparative benchmark
- [ ] Semantic loss / evidence quality diagnostics
- [ ] Stronger Korean/English discourse grammar
- [ ] Relation-aware expansion
- [ ] Persistent SQLite repository
- [ ] Optional FAISS/Qdrant/pgvector adapters

## License

Apache-2.0
