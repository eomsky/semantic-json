from semantic_json import SemanticRepository, LiteEmbedder, EvidenceRegion, HeuristicRegionModel


def test_lite_mode_is_available_without_neural_boundary_scoring():
    repo = SemanticRepository(embedder=LiteEmbedder(), region_model="lite")
    assert isinstance(repo.embedder, LiteEmbedder)
    assert isinstance(repo.region_model, HeuristicRegionModel)


def test_lite_search_returns_source_grounded_region():
    text = (
        "B기업은 현재까지 원리금을 정상적으로 상환하고 있다.\n"
        "주요 거래계약이 내년에 만료될 예정이다.\n"
        "중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다."
    )
    repo = SemanticRepository(embedder=LiteEmbedder(), region_model="lite", compatibility_threshold=0.0)
    repo.add_text(text, document_id="test")
    results = repo.search("B기업의 중장기 채무상환능력은 어떤가?", top_k=3, max_region_units=4)
    assert results
    assert isinstance(results[0], EvidenceRegion)
    assert results[0].text in text
    assert repo.verify_source(results[0])
    units = repo.search_units("B기업의 중장기 채무상환능력은 어떤가?", top_k=3)
    assert units
    assert units[0].unit_id.startswith("U")
