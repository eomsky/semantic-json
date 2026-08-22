from semantic_json import compile, SemanticRepository, LiteEmbedder


def test_default_repository_uses_lite_embedder():
    repo = SemanticRepository()
    assert isinstance(repo.embedder, LiteEmbedder)


def test_lite_search_korean_finance_synonym():
    text = """
B기업은 현재까지 원리금을 정상적으로 상환하고 있다.
다만 주요 거래계약이 내년에 만료될 예정이며,
중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다.
"""
    doc = compile(text, document_id="test")
    repo = SemanticRepository()
    repo.add(doc)
    results = repo.search("B기업의 중장기 채무상환능력은 어떤가?", top_k=3)
    assert results
    assert results[0].entity_id == "B_CORP"
    assert results[0].proposition.scope.stance == "difficult_to_conclude"
    assert results[0].proposition.scope.time == "medium_term"
