# 변경 이력

## 0.2.0a4

- 공개 문서의 기본 언어를 한국어 중심으로 정리했다. Python API, class/function name, 수식, benchmark 명칭처럼 영어가 자연스러운 부분은 그대로 유지한다.
- v0.17 Final Primary Benchmark 결과를 README와 연구 아키텍처 문서에 반영했다.
- Held-out test 1,309 queries에서 paragraph Evidence F1: Best-Tuned Dense Fixed RAG `0.2387`, Best-Tuned Reranked Fixed RAG `0.2432`, Query-Conditioned Evidence Construction `0.2694`를 기록했다.
- Reranked Fixed RAG 대비 paired bootstrap paragraph Evidence F1 차이 `+0.0262`, 95% CI `[0.0065, 0.0463]`, bootstrap `P(Δ<=0)=0.0052`를 기록했다.
- Internal unit F1은 Dense Fixed `0.1757`, Reranked Fixed `0.1804`, 제안 방법 `0.2274`였다.
- Actual context token은 각각 약 `125.9`, `127.1`, `266.7`로 달라, 제안 방법이 Fixed baseline을 Pareto-dominate한다고 주장하지 않음을 명시했다.
- Fixed-RAG strong baseline tuning 범위를 기록했다: context budget 4개, chunk length 4개, overlap 3개, dense candidate k 4개, reranker final k 5개 조합을 validation에서 반복 평가했다.
- Global checkpoint를 실제 validation NeuralBeam Evidence F1으로 선택했으며 v0.17 best epoch이 5임을 기록했다. Value correlation은 반대로 감소하여 surrogate diagnostic과 deployed search objective의 차이를 재확인했다.
- Candidate-space Oracle paragraph Evidence F1이 약 `0.603`으로 현재 system보다 크게 높아 Global selection generalization이 주요 병목임을 명시했다.
- QASPER test가 개발 과정에서 반복 관찰되었으므로 external validation 또는 fresh final-test protocol이 필요하다는 제한을 추가했다.
- Package runtime API 자체는 v0.2 계열을 유지하며 full scalar-potential Local Composer + neural Global Value/Search stack은 research preview로 유지한다.

## 0.2.0a3

- 공개 package 문서를 현재 연구 방향인 **Navigation → Local scalar-potential EvidenceRegion composition → Global neural Evidence-Set value + structured search**와 동기화했다.
- packaged v0.2 runtime은 source-grounded `SemanticRepository` / Region Compatibility API이고 scalar-potential 및 Global value/search stack은 benchmark validation 중인 research preview임을 명확히 했다.
- synthetic weighted checkpoint score를 사용하지 않고 actual deployed search Evidence F1으로 validation checkpoint를 선택하는 평가 원칙을 문서화했다.
- v0.16 phase-1 결과를 기록했다: 동일 maximum context-token budget에서 query-conditioned Evidence Construction이 validation-tuned fixed-chunk + exact-cosine vector retrieval baseline보다 높은 held-out text-evidence 성능을 보였다.
- v0.17 closing benchmark 설계에 validation-tuned reranked Fixed RAG, paired bootstrap confidence interval, actual-token Pareto analysis를 추가했다.

## 0.2.0a2

- versioned `semantic-json-transport/region-dataset/v0.1` training-data schema 추가.
- `RegionQuery`, `RegionCandidate`, structured `TeacherDecision` 추가.
- `LLMRegionTeacher`가 특정 LLM provider SDK에 의존하지 않고 decision, teacher score, confidence, reason code를 보존하도록 확장.
- source document와 query에서 adjacent pair, hard candidate, easy candidate를 자동 생성.
- query-relevant하면서 semantic하게 유사한 non-adjacent/cross-document unit에서 hard candidate mining. 최종 labeling은 teacher decision을 authoritative하게 유지.
- confidence filtering, JSONL save/load, dataset summary, schema validation 추가.
- dataset generation은 dependency-light하게 유지하고 encoder fine-tuning은 `training` extra 뒤에 유지.

## 0.2.0a1

- 핵심 문제를 precomputed semantic grammar보다 query-conditioned Region Compatibility 중심으로 재구성.
- fine-grained contiguous `SemanticUnit`과 exact source coordinates 도입.
- retrieval을 Locator와 Composer 단계로 분리.
- multilingual cross-encoder bootstrap checkpoint 기반 `RegionCompatibilityEncoder` interface 추가.
- `HeuristicRegionModel`을 explicit lite fallback으로 추가.
- compatibility score 기반 query-time boundary expansion 추가.
- distant source location은 independent EvidenceRegion으로 유지.
- structured transport를 `semantic-json-transport/context/v2`로 확장하고 unit ID와 boundary-decision trace 보존.
- source URI, line/character coordinates, SHA-256 verification, plain-text inspection, exact source recovery 유지.
- `LLMRegionTeacher`, `RegionDatasetBuilder`, `RegionEncoderTrainer` 추가.
- compatibility score를 calibrated probability와 명시적으로 구분.

## 0.1.0a7

- `SearchResult`를 `SemanticRepository.search()`의 canonical structured transport로 변경하면서 evidence region iteration/indexing 호환성 유지.
- `SearchResult.to_dict()` / `to_json()`과 versioned `semantic-json-transport/context/v1` schema 추가.
- structured transport에서 independent EvidenceRegion을 source document 단위로 grouping.
- `SearchResult.to_text()`와 backward-compatible `build_context()` 유지.
- stable region ID, optional `source_uri`, exact character/line coordinates, document SHA-256 provenance 추가.
- `locate()`, `get_source()`, `verify_source()` 추가.

## 0.1.0a6

- query-time EvidenceRegion expansion을 entity-safe하게 개선.
- context budget 적용 시 proposition boundary 보존.
- assembled region 기준 entity ID 보고.
- exclusive source offset의 end-line 계산 수정.
- `SemanticRepository.load()` 추가.
- EvidenceRegion API regression test 및 100-document demo 개선.
- CI Python 3.10-3.12 및 PyPI publication 전 test 요구.

## 0.1.0a5

- retrieval을 semantic anchor + query-time EvidenceRegion assembly 중심으로 refactor.
- Top-K를 individual proposition이 아닌 source-grounded evidence region으로 정의.
- `search_units()`를 unit-level diagnostic API로 유지.

## 0.1.0a4

- Semantic JSON Grammar v0.1 formalization.
- proposition polarity, epistemic status, temporal scope, condition, speaker, relation, provenance 분리.

## 0.1.0a3

- NumPy-only `LiteEmbedder`를 default retrieval backend로 설정.
- sentence-transformers / multilingual E5를 optional dependency로 이동.
- PyPI distribution name을 `semantic-json-transport`로 채택하고 import는 `semantic_json` 유지.

## 0.1.0a2

- 한국어 clause-level semantic scope 처리 개선.
- epistemic stance / temporal-scope regression coverage 추가.

## 0.1.0a1

- 최초 public alpha scaffold.
- LLM-free rule-based semantic compiler.
- Korean/English baseline grammar.
- Multilingual E5 embedding adapter.
- NumPy semantic repository.
- 100-document credit-review acceptance demo.
