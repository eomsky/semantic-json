# Research Architecture v0.17 — 연구 아키텍처 및 1차 Benchmark 결과

이 문서는 `semantic-json-transport` 프로젝트의 현재 연구 formulation과 v0.17 phase-1 benchmark 결과를 기록한다.

> **상태:** research preview. 현재 PyPI v0.2 runtime은 아래 전체 architecture를 stable inference API로 노출하지 않는다.

## 1. 문제 정의

일반적인 RAG는 query를 알기 전에 retrieval chunk boundary를 결정한다. 본 연구의 핵심 가설은 **원문 provenance를 유지하면서 최종 context boundary를 query-conditioned decision으로 바꿀 수 있는가**이다.

문서는 ordered primitive units로 표현한다.

```text
U1, U2, ..., Un
```

즉 indexed unit과 최종 evidence unit이 반드시 같을 필요는 없다고 본다.

## 2. 확정한 phase-1 architecture

```text
Offline indexing / hierarchy / reusable embeddings
                    ↓
              Navigation
                    ↓
       Local EvidenceRegion Composer
              Φθ(Q,R,∂R)
                    ↓
         source-contiguous basins
              R1*,...,Rm*
                    ↓
        Global Evidence-Set Value
               Vω(Q,E)
                    +
           Structured Search
                    ↓
               Evidence Set
```

세 semantic decision layer의 역할을 의도적으로 분리한다.

### Navigation

Navigation은 **어디를 볼 것인가**를 결정한다. Dense retrieval, source adjacency, semantic shortcut, document hierarchy 등을 사용할 수 있다. Long-range navigation은 새로운 anchor를 발견하지만 Local Region 자체를 non-contiguous하게 만들지는 않는다.

v0.12에서 navigation frontier 확대는 reachable evidence를 개선했다. 다만 비교 방법들의 anchor budget이 같지 않았기 때문에 semantic/hierarchical topology 자체가 개선의 유일한 원인이라고 주장하지 않는다.

### Local Composition

Local EvidenceRegion은 source-contiguous interval이다.

```text
R = [i,j]
```

Local Composer는 boundary-aware scalar potential을 학습한다.

```text
Φθ(Q,R,∂R)
```

neighboring Region edit는 potential difference로 비교한다.

```text
ΔQ(R → R') = ΦQ(R') - ΦQ(R)
```

finite Region state space에서 strict ascent는 directed cycle을 만들 수 없다. 이 formulation은 cyclic trajectory가 발생했던 independent edit-utility formulation을 대체하였다.

### Global Composition

Local scalar potential은 서로 멀리 떨어진 basin 간 globally calibrated relevance score로 가정하지 않는다. Global selection은 별도의 set-value 문제다.

```text
Vω(Q,E)
```

현재 연구 구현은 explicit unary relevance foundation과 learned set-interaction residual을 결합하고, 이 Value로 structured beam search를 안내한다.

## 3. Checkpoint 선택 원칙

과거 실험에서는 pair accuracy, correlation, MSE를 임의 계수로 합친 synthetic checkpoint score를 사용하였다. 이 방식은 폐기하였다.

현재 Global checkpoint는 실제 deployed validation decision으로 선택한다.

```text
best checkpoint = argmax validation NeuralBeam Evidence F1
```

MSE, correlation, pair accuracy는 diagnostic일 뿐 checkpoint 결정에 사용하지 않는다.

v0.17에서 validation NeuralBeam internal F1은 다음과 같았다.

```text
Epoch 1  0.1985
Epoch 2  0.2069
Epoch 3  0.2132
Epoch 4  0.2120
Epoch 5  0.2135  ← best
```

반면 Value correlation은 약 `0.394 → 0.302`로 하락했다. 이는 surrogate diagnostic과 actual search quality가 같은 방향으로 움직인다고 가정하면 안 된다는 empirical evidence다.

## 4. Scaling principle

`N`을 primitive source unit 수라고 하자.

```text
N source units
  ↓
approximate / hierarchical lookup
  ↓
k anchors
  ↓
Local potential trajectories
  ↓
m unique Region basins
  ↓
Global set search
```

재사용 가능한 `O(N)` representation work는 ingestion으로 이동한다.

Beam width `B`, candidate basin 수 `m`, maximum set depth `d`에 대해 straightforward beam implementation은 설계 수준에서 대략

```text
O(B · m · d)
```

개의 successor Value state를 평가한다. Exact subset enumeration의 `O(2^m)`과 대비된다. 이 식은 전체 neural FLOPs가 아니라 Value evaluation 횟수에 대한 근사다.

## 5. Evaluation discipline

Internal architecture diagnostic과 external benchmark metric을 분리한다.

### Internal diagnostics

- source-unit Evidence precision / recall / F1
- Boundary IoU / coverage
- candidate-space Oracle F1
- selection efficiency
- selected Region count
- actual context tokens

### QASPER-compatible text-evidence metric

선택한 EvidenceRegion을 원문 paragraph로 다시 mapping하고 현재 pipeline이 표현하는 text evidence에 대해 paragraph-set Evidence F1을 계산한다.

Complete official task protocol과 non-text evidence condition을 모두 재현하기 전에는 공개 QASPER leaderboard 수치를 apples-to-apples claim으로 사용하지 않는다.

## 6. Strong Fixed-RAG benchmark

핵심 비교 대상은 약한 fixed-window straw man이 아니다. v0.17에서는 다음 grid를 validation에서 반복 탐색하였다.

```text
context budget:       512, 1024, 2048, 4096
chunk length:         128, 256, 512, 1024
overlap:              0, 0.1, 0.2
dense candidate k:    5, 10, 20, 40
reranker final k:     1, 3, 5, 8, 10
```

조건:

```text
final_k <= dense_k
chunk_length <= context_budget
```

Dense baseline은 exact cosine vector retrieval을 사용한다. Stronger baseline은 dense candidate를 query-conditioned neural reranker로 재정렬한다.

Validation에서 선택된 reranked configuration은 모든 tested budget에서 다음과 같았다.

```text
chunk length = 128
overlap = 0.1
dense candidate k = 5
final k = 1
```

## 7. v0.17 held-out 결과

Test query 수는 1,309개다.

| Method | Paragraph Evidence F1 | Internal Unit F1 | Mean actual tokens |
|---|---:|---:|---:|
| Best-Tuned Dense Fixed RAG | 0.2387 | 0.1757 | 125.9 |
| Best-Tuned Reranked Fixed RAG | 0.2432 | 0.1804 | 127.1 |
| **Query-Conditioned Evidence Construction** | **0.2694** | **0.2274** | 266.7 |

Reranked Fixed RAG 대비 paragraph Evidence F1 차이:

```text
mean Δ = +0.0262
95% paired-bootstrap CI = [0.0065, 0.0463]
bootstrap P(Δ <= 0) = 0.0052
```

Internal unit F1 차이:

```text
mean Δ ≈ +0.0470
95% CI = [0.0300, 0.0646]
```

Dense Fixed RAG 대비 paragraph Evidence F1 차이는 약 `+0.0306`, 95% CI `[0.0103, 0.0506]`였다.

## 8. Actual-token Pareto 해석

동일 maximum budget이 동일 actual context use를 뜻하지 않는다.

```text
Dense Fixed       ≈ 125.9 tokens
Reranked Fixed    ≈ 127.1 tokens
Ours              ≈ 266.7 tokens
```

따라서 현재 방법은 더 높은 evidence quality를 얻지만 더 많은 context를 사용한다. 세 방법 모두 empirical Pareto frontier에 남았다.

현재 결과에서 주장하지 않는 것:

- 같은 actual token에서 항상 우월하다는 주장
- Fixed RAG를 Pareto-dominate한다는 주장
- 모든 dataset과 RAG 환경에서 universal superiority가 입증되었다는 주장

## 9. Oracle gap

Candidate-space Oracle paragraph Evidence F1은 약 `0.603`으로 현재 system `0.269`보다 크게 높다.

이는 candidate generation에 상당한 잠재력이 이미 존재하지만 Global Value / selection이 이를 충분히 활용하지 못하고 있음을 뜻한다. 따라서 다음 단계의 중요한 병목은 Global selection generalization이다.

## 10. Package와의 관계

현재 PyPI package가 제공하는 stable v0.2 runtime:

- `SemanticRepository`
- `SemanticUnit` / deterministic source unitization
- `EvidenceRegion` / `SearchResult` structured transport
- source coordinates / verification
- Region Compatibility model
- dataset generation / optional fine-tuning utility

Full scalar-potential Local Composer와 neural Global Value/Search architecture는 model interface, checkpoint, reproducibility requirement가 package API 수준으로 정리될 때까지 experimental이다.

## 11. 다음 검증 단계

QASPER test를 개발 과정에서 반복 관찰했으므로 phase-1 결과를 최종적인 external generalization proof로 해석하지 않는다. 다음 단계는 다음을 우선한다.

1. external dataset validation
2. final architecture ablation
3. downstream QA evaluation
4. scalability / latency benchmark
5. fresh final-test protocol
