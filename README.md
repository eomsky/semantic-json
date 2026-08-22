# Semantic JSON Transport

> **Find the location cheaply. Compose the evidence for the query. Transport the original source.**

Semantic JSON Transport builds **query-conditioned EvidenceRegions** from long documents. Documents are not pre-cut into final retrieval chunks.

## v0.2 core idea

The central operation is:

```text
region_compatibility(query, left_span, right_span) -> compatibility score
```

A document is represented as small, contiguous, source-grounded `SemanticUnit`s. A locator retrieves candidate units. A Region Compatibility Encoder then decides at query time whether neighboring source spans should travel together in the same EvidenceRegion.

```text
Long documents
    ↓
Fine-grained SemanticUnits + source coordinates
    ↓
Locator embedding index
    ↓
Query → candidate anchors
    ↓
Region Compatibility Encoder
    ↓
Query-conditioned boundary search
    ↓
Independent EvidenceRegions
    ↓
Structured JSON / plain text
    ↓
Downstream LLM
```

The indexed unit is **not** the final context. Final context is reconstructed from the original source after the query is known.

## Installation

```bash
pip install semantic-json-transport
```

Version `0.2.0a1` makes the encoder runtime part of the default installation and is CPU-capable. The current default checkpoint is a multilingual cross-encoder **bootstrap checkpoint**. It establishes and exercises the Region Compatibility interface, but it is not presented as a calibrated, task-specific Semantic JSON checkpoint. A dedicated checkpoint must be trained and benchmarked before that claim is made.

For a dependency-light boundary baseline, the package also keeps a lite mode.

## Quick start

```python
from semantic_json import SemanticRepository

text = """
B기업은 주요 거래처와 공급계약을 체결하고 있다.
해당 계약은 내년 말 만료될 예정이다.
해당 거래처 매출 의존도는 높은 수준이다.
중장기 상환능력을 낙관하기 어렵다.
"""

repo = SemanticRepository()
repo.add_text(
    text,
    document_id="company_b",
    source_uri="documents/company_b.txt",
)

result = repo.search("B기업의 중장기 상환능력", top_k=5)

print(result.to_json())   # canonical structured transport
print(result.to_text())   # human-readable retrieval inspection
```

Lite fallback:

```python
from semantic_json import SemanticRepository, LiteEmbedder

repo = SemanticRepository(
    embedder=LiteEmbedder(),
    region_model="lite",
)
```

## SemanticUnit

`SemanticUnit` is a fine-grained, contiguous source span. It may be smaller than a sentence. It is a retrieval/composition primitive, not a final chunk and not a rewritten summary.

```text
Original source
    ↓
U1 | U2 | U3 | ... | Un
```

Every unit retains exact source coordinates. The default `FineGrainedUnitizer` is deterministic and replaceable so sentence-, clause-, proposition-, tokenizer-, or learned unitizers can be compared without changing the retrieval/transport contract.

## Locator + Composer

### Stage 1 — Locator

The locator uses inexpensive embedding search to identify candidate source locations.

```text
Query → embedding retrieval → U37, U91, U240 ...
```

### Stage 2 — Composer

For each candidate location, the Region Compatibility Model evaluates neighboring spans in the context of the query.

```text
compatibility(query, left_span, right_span)
```

The boundary search expands while compatibility remains high enough. Distant locations remain independent EvidenceRegions rather than becoming one giant chunk.

Compatibility outputs are called **scores**, not probabilities, unless a model has been explicitly calibrated.

## Canonical structured transport

Search returns `SearchResult`, whose canonical serialization is JSON.

```json
{
  "schema": "semantic-json-transport/context/v2",
  "query": "B기업의 중장기 상환능력",
  "documents": [
    {
      "document_id": "company_b",
      "regions": [
        {
          "region_id": "company_b:R1",
          "score": 0.87,
          "source": {
            "start_char": 42,
            "end_char": 181,
            "start_line": 2,
            "end_line": 4,
            "document_sha256": "..."
          },
          "anchors": ["U4"],
          "units": ["U2", "U3", "U4"],
          "boundaries": [
            {
              "left_unit_id": "U2",
              "right_unit_id": "U3",
              "compatibility_score": 0.81,
              "included": true
            }
          ],
          "text": "원문 그대로..."
        }
      ]
    }
  ]
}
```

Plain text remains a first-class mode for retrieval inspection:

```python
print(result.to_text())
```

## Source provenance and verification

The source is authoritative. Every EvidenceRegion can be located and verified against the original document.

```python
region = result[0]
repo.locate(region)
repo.get_source(region)
repo.get_source(region, context_before=500, context_after=500)
repo.verify_source(region)
```

The core invariant is:

```python
document.text[region.start_char:region.end_char] == region.text
```

## Bring your own LLM teacher

An LLM can be used **offline as a judge**, never as a mandatory production dependency. The teacher labels whether adjacent spans should be transported together for a query; those examples can fine-tune the small Region Compatibility Encoder.

```python
from semantic_json import (
    LLMRegionTeacher,
    RegionDatasetBuilder,
    RegionEncoderTrainer,
    DEFAULT_REGION_MODEL,
)

teacher = LLMRegionTeacher(my_judge_callable, name="internal-llm")
builder = RegionDatasetBuilder(teacher)

examples = builder.label_pairs([
    (query, left_span, right_span),
])

trainer = RegionEncoderTrainer(DEFAULT_REGION_MODEL)
trainer.fit(examples, output_path="./my-region-encoder")
```

Use the trained encoder in production:

```python
from semantic_json import RegionCompatibilityEncoder, SemanticRepository

region_model = RegionCompatibilityEncoder("./my-region-encoder")
repo = SemanticRepository(region_model=region_model)
```

Fine-tuning is always explicit. Semantic JSON Transport never silently modifies a production model at runtime.

## Backward compatibility

The v0.1 `compile()` / `SemanticDocument` grammar APIs remain available during the alpha transition. `SemanticRepository.add(SemanticDocument)` converts proposition source spans into v0.2 source units. New applications should prefer `add_text()`.

## v0.2 design principles

- No final fixed chunks at ingestion time.
- Fine-grained source units increase retrieval/composition resolution.
- Region boundaries are query-conditioned.
- The Region Compatibility Model is pluggable.
- Compatibility score is not called probability without calibration.
- Distant evidence remains in independent EvidenceRegions.
- Original source text and provenance remain authoritative.
- Structured JSON is canonical transport; plain text is first-class inspection.
- Entity, relation, discourse, layout, and other structure are optional evidence signals for compatibility—not the project objective themselves.
- A user-provided LLM may teach a small encoder offline; production inference stays encoder-based.

## Evaluation direction

The v0.2 benchmark should separate locator quality from composer quality and compare against conventional fixed-chunk RAG using the same retrieval backbone where possible.

Primary metrics include Evidence Recall, Evidence Precision, Boundary IoU, token efficiency, contamination rate, latency, memory, and downstream answer quality.

## License

Apache-2.0
