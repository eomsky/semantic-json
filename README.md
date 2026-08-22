# Semantic JSON

> **LLM이 읽기 전에, 의미를 보존합니다.**  
> **Preserve meaning before the LLM sees it.**

Semantic JSON은 장문의 자연어 문서를 검색하고 LLM 컨텍스트로 전달하는 과정에서 **원문의 의미와 논리구조 손실을 최소화**하기 위한 경량 semantic transport layer입니다. 생성형 LLM이나 GPU를 필수로 요구하지 않으며, CPU 기반 semantic compiler와 다국어 임베딩 검색을 결합합니다.

Semantic JSON is a lightweight semantic transport layer designed to **minimize loss of meaning and logical structure** when long-form natural-language documents are retrieved and prepared for LLM input. A generative LLM and GPU are not required; the default architecture combines a CPU-friendly semantic compiler with multilingual embeddings.

**핵심 원칙 / Core principle**

> 임베딩은 관련된 의미를 찾고, Semantic JSON은 그 의미가 실제로 무엇을 말했는지 보존합니다.  
> Embeddings find related meaning; Semantic JSON preserves what the source actually says.

## 특징 / Features

- 생성형 LLM 불필요 / No generative LLM required
- GPU 불필요 / No GPU required
- 원문 source span 보존 / Source-span provenance
- entity / negation / modality / time / condition / attribution 보존을 지향 / Designed to preserve entity, negation, modality, time, condition, and attribution
- 다국어 임베딩 기반 유사어 검색 / Multilingual embedding retrieval across different wording
- Vector DB 없이 NumPy 기반 검색 가능 / NumPy search without a vector database
- LLM-ready context builder / LLM-ready context construction

## 설치 / Installation

PyPI 배포명은 `semantic-json-transport`이며 Python import 이름은 `semantic_json`입니다.  
The PyPI distribution is `semantic-json-transport`; the Python import remains `semantic_json`.

```bash
pip install semantic-json-transport
```

의미 기반 검색까지 사용하려면 / For semantic embedding search:

```bash
pip install "semantic-json-transport[search]"
```

> 현재 버전은 `0.1.0a1` alpha입니다. API와 grammar는 변경될 수 있습니다.  
> This is the `0.1.0a1` alpha. The API and semantic grammar may change.

## 빠르게 시작하기 / Quick Start

```python
from semantic_json import compile

text = """
A기업은 현재까지 원리금을 정상적으로 상환하고 있다.
다만 주요 거래계약이 내년에 만료될 예정이며,
중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다.
"""

# 장문의 원문을 의미 구조로 변환 (Compile source text into semantic structure)
doc = compile(text, document_id="company_a")
print(doc.to_json())
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

```python
from pathlib import Path
from semantic_json import compile, SemanticRepository

repo = SemanticRepository()

# 100개 문서 적재 (Compile and add 100 documents)
for path in Path("examples/credit_review_100docs/documents").glob("*.txt"):
    repo.add(compile(path.read_text(encoding="utf-8"), document_id=path.stem))

# 의미 기반 검색 (Semantic search)
results = repo.search(
    "B기업의 재무상태와 상환능력에 관련된 내용을 찾아줘.",
    top_k=20,
)

# LLM 입력 컨텍스트 생성 (Build LLM-ready context)
context = repo.build_context(results)
print(context)
```

`doc_003~100`은 저장소에 모두 커밋하지 않고 `generate_documents.py`로 재현 가능하게 생성합니다.

`doc_003~100` are generated reproducibly by `generate_documents.py` instead of being committed as repetitive fixtures.

## 아키텍처 / Architecture

```text
100 long-form .txt documents
             │
             ▼
     LLM-free Semantic Compiler
             │
             ▼
       SemanticDocument
   ┌─────────┼───────────┐
   │ entity  │ scope     │ provenance
   │ claim   │ relations │ source span
   └─────────┼───────────┘
             │
             ▼
     proposition search text
             │
             ▼
   multilingual CPU embeddings
             │
             ▼
      SemanticRepository
             │
      query → candidates
             │
             ▼
    semantic/entity filtering
             │
             ▼
      source span recovery
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

현재 alpha는 긴 **plain text**만 대상으로 합니다. 표, 이미지, PDF layout은 아직 처리하지 않습니다. Rule-based semantic compiler 역시 완전한 자연어 이해 시스템이 아니며 명시적 entity 전환, uncertainty, modality, condition 등의 보수적인 패턴부터 지원합니다.

The current alpha targets long-form **plain text** only. Tables, images, and PDF layout are out of scope for now. The rule-based compiler is not a complete natural-language understanding system; it starts with conservative support for explicit entity switching, uncertainty, modality, and conditions.

## 로드맵 / Roadmap

- [x] Long plain-text → SemanticDocument
- [x] Source-span provenance
- [x] Korean / English rule grammar baseline
- [x] Multilingual embedding adapter
- [x] NumPy SemanticRepository
- [x] 100-document credit-review acceptance demo
- [ ] Semantic loss diagnostics
- [ ] Stronger Korean/English discourse grammar
- [ ] Relation expansion during retrieval
- [ ] Persistent SQLite repository
- [ ] Optional FAISS/Qdrant/pgvector adapters

## 라이선스 / License

Apache-2.0
