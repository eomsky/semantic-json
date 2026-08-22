# Query-time Evidence Regions

이 문서는 Semantic JSON Transport의 retrieval 목표를 정의합니다.

Semantic JSON Transport should not require long documents to be permanently divided into fixed chunks before a query exists. The compiler creates small source-grounded semantic units. Retrieval finds relevant semantic anchors, then assembles source text around those anchors into evidence regions at query time.

```text
Long documents
  -> semantic units + source coordinates
  -> semantic index
  -> query
  -> relevant anchors
  -> nearby/related anchors are merged
  -> source region expansion
  -> Top-K EvidenceRegion
```

`top_k=5` therefore means five evidence regions, not five individual sentences.

각 `EvidenceRegion`은 최소한 다음 정보를 반환합니다.

- `document_id`
- `score`
- `start_char`, `end_char`
- `start_line`, `end_line`
- `text`
- `anchor_proposition_ids`
- `entity_ids`

문법 적용 단위와 최종 반환 단위는 동일할 필요가 없습니다. 문법은 작은 semantic unit에 적용하고, 최종 context/chunk는 질의 시점에 동적으로 조립합니다.

The grammar unit and retrieval unit do not need to be identical. Grammar is applied to small semantic units; the final context is assembled dynamically after the query arrives.
