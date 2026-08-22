from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
from .schemas import SemanticDocument, Proposition
from .embeddings import LiteEmbedder
from .compiler import entity_mentions

@dataclass
class SemanticMatch:
    document_id: str
    proposition_id: str
    entity_id: str
    score: float
    proposition: Proposition

class SemanticRepository:
    """벡터 DB 없이 NumPy로 동작하는 저장소 (In-memory NumPy semantic repository)."""
    def __init__(self, *, embedder=None):
        # 기본값은 외부 모델 다운로드가 없는 LiteEmbedder
        # (Default backend is LiteEmbedder with no external model download.)
        self.documents={}; self.embedder=embedder or LiteEmbedder(); self._records=[]; self._matrix=None

    def add(self,doc: SemanticDocument) -> None:
        self.documents[doc.document_id]=doc; self._matrix=None

    def _search_text(self,doc,p):
        aliases=" ".join(doc.entities.get(p.entity_id,{}).get("aliases",[])); s=p.scope
        return f"{aliases} {p.claim} time={s.time} stance={s.stance} polarity={s.polarity} speaker={s.speaker} condition={s.condition}"

    def build_index(self):
        self._records=[]; texts=[]
        for doc in self.documents.values():
            for p in doc.propositions:
                self._records.append((doc,p)); texts.append(self._search_text(doc,p))
        self._matrix=self.embedder.encode_passages(texts) if texts else np.empty((0,0))

    def search(self,query: str, *, top_k: int=10, entity_filter: bool=True):
        if self._matrix is None: self.build_index()
        if not self._records: return []
        scores=self._matrix @ self.embedder.encode_query(query)
        q_entities={eid for _,eid in entity_mentions(query)}; candidates=[]
        for idx,score in enumerate(scores.tolist()):
            doc,p=self._records[idx]
            if entity_filter and q_entities and p.entity_id not in q_entities: continue
            candidates.append(SemanticMatch(doc.document_id,p.id,p.entity_id,float(score),p))
        return sorted(candidates,key=lambda x:x.score,reverse=True)[:top_k]

    def build_context(self,results, *, include_source: bool=True) -> str:
        lines=["=== SEMANTIC EVIDENCE ==="]
        for r in results:
            p=r.proposition; lines += [f"[{r.document_id}/{p.id}] entity={p.entity_id}",f"claim: {p.claim}",f"scope: stance={p.scope.stance}, time={p.scope.time}, polarity={p.scope.polarity}"]
            if include_source: lines.append(f"source: {p.source.text}")
            lines.append("")
        return "\n".join(lines).strip()

    def save(self,path: str):
        with open(path,"w",encoding="utf-8") as f: json.dump([d.to_dict() for d in self.documents.values()],f,ensure_ascii=False,indent=2)
