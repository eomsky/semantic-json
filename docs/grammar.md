# Semantic JSON Grammar v0.1

> 의미를 하나의 점수나 라벨로 축약하지 않고, 서로 다른 의미 축을 분리해 보존합니다.  
> Preserve distinct semantic dimensions instead of collapsing meaning into a single label.

Semantic JSON Transport의 핵심 목적은 자연어를 대신 판단하는 것이 아니라, **원문의 논리구조가 retrieval과 LLM context construction 과정에서 파괴되지 않도록 운반하는 것**입니다.

The purpose of Semantic JSON Transport is not to make decisions in place of an LLM. It is to **transport the logical structure of source text without silently destroying it during retrieval and LLM context construction**.

## Proposition

각 proposition은 최소한 다음 축을 독립적으로 가집니다. / Each proposition keeps the following dimensions independent.

```json
{
  "entity_id": "B_CORP",
  "claim": "중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다",
  "scope": {
    "proposition_polarity": "affirmative",
    "epistemic_status": "difficult_to_conclude",
    "temporal_scope": "medium_term",
    "condition": "",
    "speaker": ""
  }
}
```

## proposition_polarity
명제 내용 자체가 긍정형인지 부정형인지를 나타냅니다. **좋다/나쁘다 같은 평가 방향이 아닙니다.**  
Describes whether the proposition itself is affirmative or negated. It does **not** mean favorable or unfavorable.

- `affirmative`
- `negative`

## epistemic_status
작성자가 그 명제를 어느 정도 확정적으로 제시하는지를 나타냅니다.  
Represents how strongly the source commits to the proposition.

- `asserted`
- `possible`
- `uncertain`
- `difficult_to_conclude`

`affirmative + difficult_to_conclude`는 모순이 아닙니다. 첫 값은 명제의 극성, 두 번째 값은 그 명제에 대한 인식적 확신도를 나타냅니다.  
`affirmative + difficult_to_conclude` is not contradictory: the first is proposition polarity and the second is epistemic commitment.

## temporal_scope
명제가 적용되는 시간 범위를 나타냅니다. / Describes the temporal scope.

- `current`
- `future`
- `medium_term`
- empty when unspecified

## condition
명제가 특정 조건 아래에서만 성립하는 경우 원문의 조건 표지를 보존합니다.  
Preserves an explicit condition under which the proposition applies.

## speaker
회사, 은행 심사자 등 명시적 attribution을 보존합니다.  
Preserves explicit attribution such as company management or a bank reviewer.

- `company`
- `bank`
- empty when unspecified

## relation
원문에 명시된 proposition 간 관계를 보수적으로 기록합니다.  
Relations are recorded conservatively, prioritizing explicitly marked relations.

- `contrast`
- `causal_or_consequential`
- `condition`

## provenance
모든 proposition은 원문의 `start`, `end`, `text`를 보존해야 합니다. Semantic JSON은 원문을 대체하는 요약이 아니라 **원문으로 되돌아갈 수 있는 semantic index**를 지향합니다.  
Every proposition retains source provenance. Semantic JSON is a **semantic index that can always return to the source**, not a replacement summary.

## Backward compatibility
v0.1.0a4부터 canonical field는 `proposition_polarity`, `epistemic_status`, `temporal_scope`입니다. 기존 Python attribute `polarity`, `stance`, `modality`, `time`은 호환 alias로 유지합니다. 새 JSON serialization에는 canonical field를 사용합니다.

From v0.1.0a4, canonical fields are `proposition_polarity`, `epistemic_status`, and `temporal_scope`. Legacy Python attributes remain compatibility aliases; new JSON serialization uses canonical names.
