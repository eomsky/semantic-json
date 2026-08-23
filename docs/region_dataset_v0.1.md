# Region Compatibility Dataset v0.1

Schema: `semantic-json-transport/region-dataset/v0.1`

The dataset teaches a Region Compatibility Encoder one narrow task:

> Given a query and two source-grounded spans, should they travel together in the same EvidenceRegion?

The labels are `same_region` and `split`. A teacher score is stored separately and must not be described as a calibrated probability unless calibration has been performed.

## Recommended data mix

A useful starting mixture is:

- normal adjacent candidates: source-neighboring units around real queries;
- hard candidates: query-relevant and semantically similar units that are non-adjacent or come from different documents;
- easy candidates: randomly separated units for basic boundary learning;
- human-reviewed ambiguous cases: especially high-impact entity/topic contamination cases.

Hard candidates are candidates, not assumed negatives. The teacher or human validator remains authoritative.

## Provider-neutral teacher

Semantic JSON Transport does not depend on any LLM SDK. Supply a callable:

```python
from semantic_json import LLMRegionTeacher


def judge(query: str, left: str, right: str):
    return {
        "label": "same_region",
        "score": 0.91,
        "confidence": "high",
        "reason_codes": ["same_evidence_chain"],
    }

teacher = LLMRegionTeacher(judge, name="my-teacher")
```

The callable may wrap a local model, an internal API, or any external LLM chosen by the user.

## Generate a dataset

```python
from semantic_json import RegionDatasetBuilder, RegionQuery

builder = RegionDatasetBuilder(teacher, random_seed=13)

examples = builder.build(
    documents={
        "doc_001": open("doc_001.txt", encoding="utf-8").read(),
        "doc_002": open("doc_002.txt", encoding="utf-8").read(),
    },
    queries=[
        RegionQuery(
            "B기업의 중장기 상환능력 관련 근거",
            query_id="Q001",
            document_id="doc_002",
        )
    ],
    adjacent_per_query=100,
    hard_candidates_per_query=20,
    easy_candidates_per_query=10,
)

builder.save_jsonl(examples, "region_dataset.jsonl")
print(builder.summarize(examples))
```

Candidate generation itself does not require an LLM provider package. `LiteEmbedder` is sufficient for default hard-candidate mining.

## Example row

```json
{
  "schema": "semantic-json-transport/region-dataset/v0.1",
  "query": "B기업의 중장기 상환능력 관련 근거",
  "left": "주요 거래계약은 내년에 만료될 예정이다.",
  "right": "해당 거래처에 대한 매출 의존도는 높은 수준이다.",
  "label": 0.91,
  "decision": "same_region",
  "teacher": "my-teacher",
  "teacher_score": 0.91,
  "confidence": "high",
  "reason_codes": ["same_evidence_chain"],
  "query_id": "Q001",
  "left_document_id": "doc_002",
  "right_document_id": "doc_002",
  "left_unit_id": "U12",
  "right_unit_id": "U13",
  "candidate_type": "adjacent",
  "metadata": {}
}
```

## Fine-tune the encoder

Dataset generation is available without installing an LLM provider SDK. Explicit encoder fine-tuning uses the training extra:

```bash
pip install "semantic-json-transport[training]"
```

```python
from semantic_json import RegionDatasetBuilder, RegionEncoderTrainer, DEFAULT_REGION_MODEL

examples = RegionDatasetBuilder.load_jsonl("region_dataset.jsonl")
trainer = RegionEncoderTrainer(DEFAULT_REGION_MODEL)
trainer.fit(
    examples,
    output_path="./my-region-encoder",
    epochs=2,
    min_teacher_confidence="medium",
)
```

Production inference then uses only the trained Region Compatibility Encoder; the teacher LLM is not required.

## Reliability rules

1. Keep source spans unchanged; never train on rewritten evidence as if it were original source.
2. Separate `decision` from `teacher_score`.
3. Do not call teacher scores probabilities unless they are calibrated.
4. Include hard cross-entity and same-topic/different-context examples.
5. Filter or human-review low-confidence teacher decisions.
6. Keep a held-out document set so train/evaluation text does not leak across splits.
