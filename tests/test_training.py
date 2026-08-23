from semantic_json import (
    DATASET_SCHEMA,
    LLMRegionTeacher,
    RegionDatasetBuilder,
    RegionQuery,
    SAME_REGION,
    SPLIT,
)


def test_llm_teacher_preserves_structured_decision_metadata():
    def judge(query, left, right):
        if "상환" in right:
            return {
                "label": "same_region",
                "score": 0.91,
                "confidence": "high",
                "reason_codes": ["same_evidence_chain"],
            }
        return {"label": "split", "score": 0.08, "confidence": "high"}

    teacher = LLMRegionTeacher(judge, name="test-teacher")
    builder = RegionDatasetBuilder(teacher)
    rows = builder.label_pairs([
        ("B기업 상환능력", "계약이 내년에 만료된다.", "상환능력 모니터링이 필요하다."),
        ("B기업 상환능력", "계약이 내년에 만료된다.", "A기업은 공장을 증설한다."),
    ])

    assert rows[0].decision == SAME_REGION
    assert rows[0].label == 0.91
    assert rows[0].confidence == "high"
    assert rows[0].reason_codes == ["same_evidence_chain"]
    assert rows[1].decision == SPLIT
    assert all(row.teacher == "test-teacher" for row in rows)


def test_candidate_generation_includes_adjacent_hard_and_easy_cases():
    documents = {
        "b": (
            "B기업은 주요 계약을 보유하고 있으나, 내년에 만료될 예정이다.\n"
            "해당 거래처 매출 비중은 높다.\n"
            "중장기 상환능력을 낙관하기 어렵다."
        ),
        "c": (
            "C기업도 주요 계약 만료를 앞두고 있다.\n"
            "C기업의 중장기 상환능력도 모니터링이 필요하다."
        ),
    }
    builder = RegionDatasetBuilder(random_seed=7)
    candidates = builder.generate_candidates(
        documents=documents,
        queries=[RegionQuery("B기업 중장기 상환능력", query_id="Q1", document_id="b")],
        adjacent_per_query=10,
        hard_candidates_per_query=2,
        easy_candidates_per_query=1,
    )
    kinds = {row.candidate_type for row in candidates}
    assert "adjacent" in kinds
    assert "hard_candidate" in kinds
    assert "easy_candidate" in kinds
    assert all(row.query_id == "Q1" for row in candidates)


def test_dataset_build_jsonl_roundtrip_and_summary(tmp_path):
    documents = {
        "doc": "B기업은 계약이 내년에 만료된다.\n중장기 상환능력 모니터링이 필요하다.\nA기업은 신규 공장을 건설한다."
    }

    def judge(query, left, right):
        same = "B기업" in left or "상환능력" in right
        return {
            "label": SAME_REGION if same else SPLIT,
            "score": 0.9 if same else 0.1,
            "confidence": "high",
        }

    builder = RegionDatasetBuilder(LLMRegionTeacher(judge), random_seed=1)
    rows = builder.build(
        documents=documents,
        queries=["B기업 상환능력"],
        hard_candidates_per_query=0,
        easy_candidates_per_query=0,
    )
    path = tmp_path / "region_dataset.jsonl"
    builder.save_jsonl(rows, path)
    loaded = builder.load_jsonl(path)
    summary = builder.summarize(loaded)

    assert loaded
    assert all(row.schema == DATASET_SCHEMA for row in loaded)
    assert summary["schema"] == DATASET_SCHEMA
    assert summary["examples"] == len(rows)
    assert sum(summary["decisions"].values()) == len(rows)


def test_low_confidence_teacher_rows_can_be_excluded():
    teacher = LLMRegionTeacher(
        lambda q, l, r: {"label": "same_region", "score": 0.55, "confidence": "low"}
    )
    builder = RegionDatasetBuilder(teacher)
    candidates = builder.generate_candidates(
        documents={"d": "첫 번째 근거이다.\n두 번째 근거이다."},
        queries=["근거"],
        hard_candidates_per_query=0,
        easy_candidates_per_query=0,
    )
    assert builder.label_candidates(candidates, include_low_confidence=False) == []
