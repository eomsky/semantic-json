from semantic_json import SemanticRepository, LiteEmbedder, EvidenceRegion


class AlwaysJoin:
    def score(self, query, left, right):
        return 1.0


class NeverJoin:
    def score(self, query, left, right):
        return 0.0


def test_query_time_composer_can_expand_from_anchor():
    text = "A기업은 정상 영업 중이다.\nB기업은 계약이 내년에 만료된다.\n매출 의존도가 높다.\n중장기 상환능력을 낙관하기 어렵다.\nC기업은 신규 공장을 건설한다."
    repo = SemanticRepository(embedder=LiteEmbedder(), region_model=AlwaysJoin())
    repo.add_text(text, document_id="doc")
    result = repo.search("B기업 중장기 상환능력", top_k=1, max_region_units=3)
    assert result
    region = result[0]
    assert isinstance(region, EvidenceRegion)
    assert len(region.unit_ids) <= 3
    assert region.text in text
    assert region.boundary_decisions
    assert repo.verify_source(region)


def test_boundary_model_can_keep_anchor_as_single_unit():
    text = "B기업의 계약이 만료된다.\n중장기 상환능력을 낙관하기 어렵다.\nA기업은 정상 영업 중이다."
    repo = SemanticRepository(embedder=LiteEmbedder(), region_model=NeverJoin())
    repo.add_text(text, document_id="doc")
    region = repo.search("중장기 상환능력", top_k=1)[0]
    assert len(region.unit_ids) == 1
    assert region.text in text


def test_structured_transport_exposes_boundary_trace_and_source():
    text = "B기업은 계약이 내년에 만료된다.\n매출 의존도가 높다.\n상환능력 모니터링이 필요하다."
    repo = SemanticRepository(embedder=LiteEmbedder(), region_model=AlwaysJoin())
    repo.add_text(text, document_id="d", source_uri="documents/d.txt")
    result = repo.search("B기업 상환능력", top_k=1, max_region_units=2)
    payload = result.to_dict()
    assert payload["schema"] == "semantic-json-transport/context/v2"
    r = payload["documents"][0]["regions"][0]
    assert r["source"]["uri"] == "documents/d.txt"
    assert r["units"]
    assert repo.get_source(result[0]) == result[0].text
    assert repo.verify_source(result[0])
