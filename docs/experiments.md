# 실험에서 배운 점 / What We Learned

Semantic JSON은 여러 내부 프로토타입을 거치며 발전했습니다. 아래 관찰은 소규모 개발 benchmark에서 얻은 것이며 일반화된 성능 보장을 의미하지 않습니다.

Semantic JSON evolved through a series of internal prototypes. These observations come from small development benchmarks and should not be interpreted as general performance guarantees.

- Plain JSON은 proposition을 남기면서도 relation과 scope를 잃을 수 있었습니다. / Plain JSON could preserve propositions while losing relations and scope.
- Raw retrieval은 분산 근거가 길어질수록 필요한 관계를 놓치는 경우가 있었습니다. / Raw retrieval sometimes missed relations when evidence was distributed.
- Source provenance와 논리 연산자를 함께 보존하는 것이 중요했습니다. / Preserving provenance together with logical operators was important.
- 자연스러운 `A → 잠깐 B → 다시 A` 담화에서는 entity contamination보다 scope preservation이 더 어려운 문제로 나타났습니다. / In natural `A → brief B → resume A` discourse, scope preservation appeared harder than entity contamination.
- 공개 alpha의 기본 방향은 생성형 LLM 없이 CPU에서 동작하는 compiler + multilingual embedding search입니다. / The public alpha targets an LLM-free CPU compiler plus multilingual embedding search.

현재 rule-based compiler는 완전한 의미 분석기가 아닙니다. 명시적 entity 전환, uncertainty, modality, condition부터 보수적으로 처리합니다.

The current rule-based compiler is not a complete semantic parser. It intentionally begins with conservative handling of explicit entity switching, uncertainty, modality, and conditions.
