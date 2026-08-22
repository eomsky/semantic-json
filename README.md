# Semantic JSON Transport

> **LLM이 읽기 전에, 의미를 보존합니다.**  
> **Preserve meaning before the LLM sees it.**

Semantic JSON Transport는 장문의 자연어 문서를 검색하고 LLM 컨텍스트로 전달하는 과정에서 **원문의 의미와 논리구조 손실을 최소화**하기 위한 경량 semantic transport layer입니다. 기본 설치는 생성형 LLM, GPU, PyTorch, Transformers 또는 외부 모델 다운로드를 요구하지 않습니다.

Semantic JSON Transport is a lightweight semantic transport layer designed to **minimize loss of meaning and logical structure** when long-form natural-language documents are retrieved and prepared for LLM input. The default installation requires no generative LLM, GPU, PyTorch, Transformers, or external model download.

**핵심 원칙 / Core principle**

> 검색 표현은 관련된 의미를 찾고, Semantic JSON은 그 의미가 실제로 무엇을 말했는지 보존합니다.  
> Retrieval representations find related meaning; Semantic JSON preserves what the source actually says.

## 특징 / Features

- 생성형 LLM 불필요 / No generative LLM required
- GPU 불필요 / No GPU required
- 기본 검색에 PyTorch / Transformers 불필요 / No PyTorch or Transformers for default search
- 외부 embedding model 다운로드 없이 즉시 검색 / Search without downloading an embedding model
- 원문 source span 보존 / Source-span provenance
- entity / negation / modality / time / condition / attribution 보존을 지향 / Designed to preserve entity, negation, modality, time, condition, and attribution
- 한·영 의미 정규화 + hashing 기반 LiteEmbedder / Korean-English semantic normalization + hashing-based LiteEmbedder
- 선택형 multilingual E5 / sentence-transformers backend / Optional multilingual E5 / sentence-transformers backend
- Vector DB 없이 NumPy 기반 검색 / NumPy search without a vector database
- LLM-ready context builder / LLM-ready context construction

## 설치 / Installation

PyPI 배포명은 `semantic-json-transport`이며 Python import 이름은 `semantic_json`입니다.  
The PyPI distribution is `semantic-json-transport`; the Python import remains `semantic_json`.

```bash
pip install semantic-json-transport
```

이 설치만으로 compile과 기본 semantic search가 모두 동작합니다.  
This installation alone supports both compilation and the default semantic search.

더 높은 범용 semantic similarity가 필요하면 선택적으로 sentence-transformers backend를 설치할 수 있습니다.

For stronger general-purpose semantic similarity, optionally install the sentence-transformers backend:

```bash
pip install "semantic-json-transport[transformers]"
```

> 현재 버전은 `0.1.0a3` alpha입니다. API와 grammar는 변경될 수 있습니다.  
> This is the `0.1.0a3` alpha. The API and semantic grammar may change.

## 빠르게 시작하기 / Quick Start

```python
from semantic_json import compile, SemanticRepository

text = """
B기업은 현재까지 원리금을 정상적으로 상환하고 있다.
다만 주요 거래계약이 내년에 만료될 예정이며,
중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다.
"""

# 장문의 원문을 의미 구조로 변환 (Compile source text into semantic structure)
doc = compile(text, document_id="company_b")

# 기본 저장소는 LiteEmbedder를 사용 (Default repository uses LiteEmbedder)
repo = SemanticRepository()
repo.add(doc)

# 별도 모델 다운로드 없는 의미 기반 검색 (Semantic search without a model download)
results = repo.search("B기업의 중장기 채무상환능력은 어떤가?")
print(repo.build_context(results))
```

고성능 optional backend를 명시적으로 선택할 수도 있습니다.

You can explicitly select the optional higher-quality backend:

```python
from semantic_json import SemanticRepository, MultilingualE5Embedder

repo = SemanticRepository(embedder=MultilingualE5Embedder())
```

## 100개 여신심사 문서에서 숨어 있는 B기업 찾기  
## Finding B Corp. Hidden Across 100 Credit-Review Documents

```text
doc_001
A기업 이야기...
A기업 이야기...
    B기업 이야기  ← 찾아야 함
    B기업 이야기  ← 찾아야 함
다시 A기업 이야기...

doc_002
B기업 이야기      ← 찾아야 함
B기업 이야기      ← 찾아야 함
    C기업 이야기
다시 B기업 이야기  ← 찾아야 함

doc_003 ~ doc_100
B기업과 무관한 문서
```

목표는 단순히 `doc_001`, `doc_002`를 찾는 것이 아닙니다. **두 문서 안에서 B기업에 귀속되는 evidence를 모두 회수하면서 A/C기업 정보가 섞이지 않아야 합니다.**

The goal is not merely to retrieve `doc_001` and `doc_002`. The system should **recover the B Corp. evidence inside both documents without contaminating it with A Corp. or C Corp. facts.**

`doc_003~100`은 `generate_documents.py`로 재현 가능하게 생성합니다.

`doc_003~100` are generated reproducibly by `generate_documents.py`.

## 검색 backend / Retrieval Backends

### LiteEmbedder — 기본값 / Default

`LiteEmbedder`는 NumPy와 deterministic hashing을 사용하며 작은 한·영 의미 정규화 사전을 포함합니다. 설치가 매우 가볍고 외부 모델 다운로드가 없다는 장점이 있지만, 대규모 neural embedding model과 동일한 범용 의미 이해 성능을 주장하지 않습니다.

`LiteEmbedder` uses NumPy, deterministic hashing, and a small Korean-English semantic normalization lexicon. It is lightweight and requires no model download, but it is not intended to match the general semantic capability of a large neural embedding model.

### MultilingualE5Embedder — 선택형 / Optional

`MultilingualE5Embedder`는 `sentence-transformers`와 `intfloat/multilingual-e5-small`을 사용합니다. 더 무겁지만 범용 semantic similarity가 중요한 경우 선택할 수 있습니다.

`MultilingualE5Embedder` uses `sentence-transformers` and `intfloat/multilingual-e5-small`. It is heavier but remains available when stronger general-purpose semantic similarity is needed.

## 아키텍처 / Architecture

```text
Long-form .txt documents
          │
          ▼
 LLM-free Semantic Compiler
          │
          ▼
    SemanticDocument
 entity / scope / relation / provenance
          │
          ▼
 Retrieval representation
          │
    ┌─────┴──────────────┐
    ▼                    ▼
 LiteEmbedder       Multilingual E5
 (default)            (optional)
 NumPy only          Transformers
    │                    │
    └────────┬───────────┘
             ▼
      SemanticRepository
             │
             ▼
      source recovery
             │
             ▼
       LLM-ready context
             │
             ▼
            LLM
```

LLM은 Semantic JSON을 생성하기 위한 필수 dependency가 아닙니다. **최종 reasoning/generation 단계의 소비자**입니다.

The LLM is not a required dependency for creating Semantic JSON. It is the **downstream consumer for final reasoning or generation**.

## 현재 한계 / Current Limitations

현재 alpha는 긴 **plain text**만 대상으로 합니다. 표, 이미지, PDF layout은 아직 처리하지 않습니다. Rule-based semantic compiler와 LiteEmbedder 역시 완전한 자연어 이해 시스템이 아닙니다. 특히 LiteEmbedder의 cross-lingual recall은 현재 내장된 semantic normalization 범위에 영향을 받습니다.

The current alpha targets long-form **plain text** only. Tables, images, and PDF layout are out of scope for now. Neither the rule-based compiler nor LiteEmbedder is a complete natural-language understanding system. LiteEmbedder's cross-lingual recall currently depends in part on its built-in semantic normalization coverage.

## 로드맵 / Roadmap

- [x] Long plain-text → SemanticDocument
- [x] Source-span provenance
- [x] Korean / English rule grammar baseline
- [x] NumPy-only LiteEmbedder
- [x] Optional sentence-transformers backend
- [x] NumPy SemanticRepository
- [x] 100-document credit-review acceptance demo
- [ ] Semantic loss diagnostics
- [ ] Lite vs neural retrieval benchmark
- [ ] Stronger Korean/English discourse grammar
- [ ] Relation expansion during retrieval
- [ ] Persistent SQLite repository
- [ ] Optional FAISS/Qdrant/pgvector adapters

## 라이선스 / License

Apache-2.0
