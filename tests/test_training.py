from semantic_json import LLMRegionTeacher, RegionDatasetBuilder


def test_llm_teacher_builds_region_training_examples_without_provider_dependency():
    def judge(query, left, right):
        return {"score": 0.9 if "상환" in right else 0.1}

    teacher = LLMRegionTeacher(judge, name="test-teacher")
    builder = RegionDatasetBuilder(teacher)
    rows = builder.label_pairs([
        ("B기업 상환능력", "계약이 내년에 만료된다.", "상환능력 모니터링이 필요하다."),
        ("B기업 상환능력", "계약이 내년에 만료된다.", "A기업은 공장을 증설한다."),
    ])

    assert len(rows) == 2
    assert rows[0].label == 0.9
    assert rows[1].label == 0.1
    assert all(row.teacher == "test-teacher" for row in rows)
