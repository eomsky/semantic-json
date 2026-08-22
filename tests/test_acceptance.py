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
