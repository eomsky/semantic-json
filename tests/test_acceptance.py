from pathlib import Path
import sys
from semantic_json import compile

ROOT=Path(__file__).parents[1]
sys.path.insert(0,str(ROOT/"examples"/"credit_review_100docs"))
from generate_documents import generate_irrelevant_documents


def test_hidden_entity_attribution():
    generate_irrelevant_documents()
    base=ROOT/"examples"/"credit_review_100docs"/"documents"
    d1=compile((base/"doc_001.txt").read_text(encoding="utf-8"),document_id="doc_001")
    d2=compile((base/"doc_002.txt").read_text(encoding="utf-8"),document_id="doc_002")
    b1=[p for p in d1.propositions if p.entity_id=="B_CORP"]
    b2=[p for p in d2.propositions if p.entity_id=="B_CORP"]
    assert b1
    assert b2
    assert any("정상적으로 상환" in p.claim for p in b1)
    assert any("repayment capacity" in p.claim.lower() for p in b2)


def test_no_b_in_irrelevant_docs():
    generate_irrelevant_documents()
    base=ROOT/"examples"/"credit_review_100docs"/"documents"
    for i in range(3,101):
        p=base/f"doc_{i:03d}.txt"
        doc=compile(p.read_text(encoding="utf-8"),document_id=p.stem)
        assert all(x.entity_id!="B_CORP" for x in doc.propositions)


def test_korean_clause_level_epistemic_scope():
    text="""B기업은 현재까지 원리금을 정상적으로 상환하고 있다.

다만 주요 거래계약이 내년에 만료될 예정이며, 중장기적으로 현재의 상환능력이 유지된다고 단정하기는 어렵다."""
    doc=compile(text,document_id="scope_regression")
    b=[p for p in doc.propositions if p.entity_id=="B_CORP"]

    # 거래계약 만료 전망과 중장기 상환능력 판단은 별도 proposition이어야 한다.
    # (Contract expiry and medium-term repayment judgement must be separate propositions.)
    expiry=next(p for p in b if "거래계약" in p.claim)
    repayment=next(p for p in b if "중장기" in p.claim)

    assert expiry.id != repayment.id
    assert expiry.scope.time == "future"
    assert expiry.scope.stance == "asserted"
    assert repayment.scope.time == "medium_term"
    assert repayment.scope.stance == "difficult_to_conclude"
    assert repayment.importance == "core"
    assert expiry.source.text in text
    assert repayment.source.text in text
