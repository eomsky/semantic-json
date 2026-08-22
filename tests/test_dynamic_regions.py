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
    assert results[0].entity_ids == ["B_CORP"]


def test_top_k_means_regions_not_sentences():
    text = (
        "B기업은 원리금을 정상 상환하고 있다.\n"
        "중장기 상환능력 유지 여부는 불확실하다.\n"
        "추가 모니터링이 필요하다.\n"
        + ("A기업은 정상 영업 중이다.\n" * 20)
        + "B기업의 유동성은 충분하다.\n"
        "다만 향후 차입부담 확대 가능성이 있다.\n"
    )
    repo = SemanticRepository()
    repo.add(compile(text, document_id="doc"))
    regions = repo.search("B기업의 상환능력과 차입부담", top_k=2, before=1, after=1)
    assert len(regions) <= 2
    assert all(isinstance(r, EvidenceRegion) for r in regions)


def test_search_units_remains_available_for_debugging():
    repo = SemanticRepository()
    repo.add(compile("B기업은 원리금을 정상적으로 상환하고 있다.", document_id="d"))
    units = repo.search_units("B기업 상환", top_k=5)
    assert units and units[0].proposition_id.startswith("P")


def test_entity_safe_expansion_stops_at_unrelated_entity_boundary():
    text = (
        "A기업은 매출이 증가하였다.\n\n"
        "A기업은 현금흐름이 안정적이다.\n\n"
        "B기업은 상환능력이 불확실하다.\n\n"
        "C기업은 유동성이 양호하다.\n\n"
        "C기업은 부채비율이 낮다."
    )
    repo = SemanticRepository()
    repo.add(compile(text, document_id="mixed"))

    regions = repo.search("B기업 상환능력", top_k=3, before=2, after=2)
    assert regions
    b_region = regions[0]
    assert "B기업은 상환능력이 불확실하다" in b_region.text
    assert "A기업" not in b_region.text
    assert "C기업" not in b_region.text
    assert b_region.entity_ids == ["B_CORP"]


def test_entity_resumption_yields_separate_regions_across_other_entity():
    text = (
        "B기업은 상환능력이 양호하다.\n\n"
        "B기업은 현금흐름도 안정적이다.\n\n"
        "C기업은 별도 사업을 영위한다.\n\n"
        "C기업은 매출이 증가하였다.\n\n"
        "B기업은 향후 차입부담 확대 가능성이 있다.\n\n"
        "B기업은 중장기 상환능력 모니터링이 필요하다."
    )
    repo = SemanticRepository()
    repo.add(compile(text, document_id="resume"))

    regions = repo.search("B기업 상환능력 차입부담", top_k=5, before=1, after=1)
    assert regions
    assert all("C기업" not in r.text for r in regions if "B_CORP" in r.entity_ids)
    assert any("상환능력이 양호하다" in r.text for r in regions)
    assert any("차입부담 확대 가능성" in r.text for r in regions)


def test_context_budget_never_slices_inside_anchor_proposition():
    anchor = "B기업은 중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다."
    text = "B기업은 정상 영업 중이다.\n\n" + anchor + "\n\nB기업은 추가 모니터링이 필요하다."
    repo = SemanticRepository()
    repo.add(compile(text, document_id="budget"))

    regions = repo.search(
        "B기업 중장기 상환능력",
        top_k=1,
        before=1,
        after=1,
        max_context_chars=20,
    )
    assert regions
    assert anchor in regions[0].text
    assert regions[0].text in text


def test_end_line_uses_last_included_character():
    text = "B기업은 상환능력이 불확실하다.\nA기업은 정상 영업 중이다."
    repo = SemanticRepository()
    repo.add(compile(text, document_id="lines"))
    region = repo.search("B기업 상환능력", top_k=1, before=0, after=0)[0]
    assert region.start_line == 1
    assert region.end_line == 1
