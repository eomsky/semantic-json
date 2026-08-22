# Changelog

## 0.1.0a7

- Make `SearchResult` the canonical structured transport returned by `SemanticRepository.search()` while preserving iteration, indexing, length, and truthiness over evidence regions.
- Add `SearchResult.to_dict()` and `to_json()` with the versioned `semantic-json-transport/context/v1` schema.
- Group multiple independent EvidenceRegions under their source document in structured transport output.
- Keep plain-text retrieval inspection through `SearchResult.to_text()` and backward-compatible `build_context()`.
- Add stable region IDs, optional `source_uri`, exact character/line coordinates, and document SHA-256 provenance.
- Add `locate()`, `get_source()`, and `verify_source()` for audit and source-grounded evidence review.
- Add regression tests for structured transport, plain-text inspection, source recovery, and tamper detection.

## 0.1.0a6

- Make query-time EvidenceRegion expansion entity-safe by default.
- Preserve proposition boundaries when enforcing context budgets; anchor propositions are never character-sliced.
- Report entity IDs from the assembled region rather than anchor metadata alone.
- Fix end-line calculation for exclusive source offsets.
- Add `SemanticRepository.load()` for save/load symmetry.
- Migrate the LiteEmbedder regression test and 100-document demo to the EvidenceRegion API.
- Add entity contamination, entity resumption, context-budget, and line-number regression tests.
- Add CI for Python 3.10-3.12 and require tests before PyPI publication.

## 0.1.0a5

- Refactor retrieval around semantic anchors and query-time EvidenceRegion assembly.
- Define Top-K as source-grounded evidence regions rather than individual propositions.
- Keep `search_units()` as the unit-level diagnostic API.

## 0.1.0a4

- Formalize Semantic JSON Grammar v0.1.
- Separate proposition polarity, epistemic status, temporal scope, condition, speaker, relations, and provenance.
- Preserve backward-compatible Python aliases for earlier alpha scopes.

## 0.1.0a3

- Make the NumPy-only `LiteEmbedder` the default retrieval backend.
- Move sentence-transformers / multilingual E5 to an optional dependency.
- Adopt `semantic-json-transport` as the PyPI distribution name while keeping `semantic_json` as the import.

## 0.1.0a2

- Improve Korean clause-level semantic scope handling.
- Add epistemic stance and temporal-scope regression coverage.

## 0.1.0a1

- Initial public alpha scaffold.
- LLM-free rule-based semantic compiler.
- Korean/English baseline grammar.
- Multilingual E5 embedding adapter.
- NumPy semantic repository.
- 100-document credit-review acceptance demo.
