from semantic_json import compile, SemanticRepository, EvidenceRegion


def test_search_returns_dynamic_evidence_regions():
    filler = "A기업은 정상 영업을 지속하고 있다.\n" * 30
    target = (
        "B기업은 현재 원리금을 정상적으로 상환하고 있다.\n"
        "다만 주요 거래계약이 내년에 만료될 예정이며, 중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다.\n"
    )
    text = filler + target + filler
    repo = SemanticRepository()
    repo.add(compile(text, document_id="doc_002"))

    results = repo.search("B기업의 중장기 채무상환능력", top_k=5, before=1, after=1)
    assert results
    assert isinstance(results[0], EvidenceRegion)
    assert results[0].document_id == "doc_002"
    assert "상환능력이 유지된다고 단정하기는 어렵다" in results[0].text
    assert results[0].start_line > 20
    assert results[0].end_line >= results[0].start_line
    assert "B_CORP" in results[0].entity_ids


def test_top_k_means_regions_not_sentences():
    text = (
        "B기업은 원리금을 정상 상환하고 있다.\n"
        "중장기 상환능력 유지 여부는 불확실하다.\n"
        "추가 모니터링이 필요하다.\n"
        + ("A기업은 정상 영업 중이다.\n" * 20)
        + "B기업의 유동성은 충분하다.\n"
        "다만 향후 차입부담 확대 가능성이 있다.\n"
    )
    repo = SemanticRepository(); repo.add(compile(text, document_id="doc"))
    regions = repo.search("B기업의 상환능력과 차입부담", top_k=2, before=1, after=1)
    assert len(regions) <= 2
    assert all(isinstance(r, EvidenceRegion) for r in regions)


def test_search_units_remains_available_for_debugging():
    repo=SemanticRepository(); repo.add(compile("B기업은 원리금을 정상적으로 상환하고 있다.",document_id="d"))
    units=repo.search_units("B기업 상환",top_k=5)
    assert units and units[0].proposition_id.startswith("P")
