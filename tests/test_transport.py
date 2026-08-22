import json

from semantic_json import compile, SemanticRepository, SearchResult, TRANSPORT_SCHEMA


def _repo():
    text = (
        "A기업은 정상 영업을 지속하고 있다.\n\n"
        "B기업은 현재 원리금을 정상적으로 상환하고 있다.\n\n"
        "B기업의 중장기 상환능력 유지 여부는 불확실하다.\n\n"
        "C기업은 별도 사업을 영위한다."
    )
    repo = SemanticRepository(region_model="lite", compatibility_threshold=0.0)
    repo.add(compile(text, document_id="doc_002", source_uri="file:///docs/doc_002.txt"))
    return repo


def test_search_returns_structured_result_with_sequence_compatibility():
    repo = _repo()
    result = repo.search("B기업 중장기 상환능력", top_k=3)
    assert isinstance(result, SearchResult)
    assert result
    assert result[0].document_id == "doc_002"
    assert list(result) == result.regions


def test_canonical_json_groups_regions_under_documents():
    repo = _repo()
    result = repo.search("B기업 중장기 상환능력", top_k=3)
    payload = result.to_dict()
    assert payload["schema"] == TRANSPORT_SCHEMA
    assert payload["query"] == "B기업 중장기 상환능력"
    assert payload["documents"][0]["document_id"] == "doc_002"
    region = payload["documents"][0]["regions"][0]
    assert region["source"]["document_id"] == "doc_002"
    assert region["source"]["uri"] == "file:///docs/doc_002.txt"
    assert region["text"]
    assert json.loads(result.to_json())["schema"] == TRANSPORT_SCHEMA


def test_plain_text_mode_exposes_retrieval_quality_and_coordinates():
    repo = _repo()
    result = repo.search("B기업 중장기 상환능력", top_k=3)
    text = result.to_text()
    assert "EVIDENCE REGIONS" in text
    assert "doc_002" in text
    assert "lines=" in text
    assert "chars=" in text
    assert result[0].text in text
    assert repo.build_context(result) == text


def test_region_can_be_located_recovered_and_verified_against_source():
    repo = _repo()
    result = repo.search("B기업 중장기 상환능력", top_k=3)
    region = result[0]
    location = repo.locate(region)
    assert location["region_id"] == region.region_id
    assert location["start_char"] == region.start_char
    assert location["end_char"] == region.end_char
    assert location["document_sha256"]
    assert repo.get_source(region) == region.text
    assert region.text in repo.get_source(region, context_before=30, context_after=30)
    assert repo.verify_source(region) is True


def test_source_verification_detects_tampered_region_text():
    repo = _repo()
    region = repo.search("B기업 상환", top_k=1)[0]
    region.text += "tampered"
    assert repo.verify_source(region) is False
