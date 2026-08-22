from pathlib import Path
from semantic_json import compile, SemanticRepository
from generate_documents import generate_irrelevant_documents

BASE=Path(__file__).parent/"documents"
generate_irrelevant_documents()
repo=SemanticRepository()

# 100개 장문 문서를 컴파일하여 저장소에 추가 (Compile and add 100 long-form documents)
for path in sorted(BASE.glob("*.txt")):
    repo.add(compile(path.read_text(encoding="utf-8"),document_id=path.stem,language="auto"))

# 한국어 질의로 한국어/영어 문서를 함께 검색 (Search Korean and English documents with a Korean query)
query="B기업의 재무상태와 상환능력에 관련된 내용을 찾아줘."
results=repo.search(query,top_k=20)
documents=sorted({r.document_id for r in results if r.entity_id=="B_CORP"})
print("Relevant documents:",documents)
print("\n--- LLM-ready context ---")
print(repo.build_context(results[:10]))
