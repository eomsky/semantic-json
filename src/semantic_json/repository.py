from __future__ import annotations
from dataclasses import dataclass
import json
import numpy as np
from .schemas import SemanticDocument, Proposition
from .embeddings import LiteEmbedder
from .compiler import entity_mentions

@dataclass
class SemanticMatch:
    """내부 검색용 semantic anchor (Internal semantic anchor)."""
    document_id: str
    proposition_id: str
    entity_id: str
    score: float
    proposition: Proposition

@dataclass
class EvidenceRegion:
    """질의 시점에 조립되는 최종 검색 단위 (Query-time assembled retrieval region)."""
    document_id: str
    score: float
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    text: str
    anchor_proposition_ids: list[str]
    entity_ids: list[str]

class SemanticRepository:
    """Semantic anchors를 검색하고 원문 evidence region을 동적으로 조립합니다."""
    def __init__(self, *, embedder=None):
        self.documents={}; self.embedder=embedder or LiteEmbedder(); self._records=[]; self._matrix=None

    def add(self, doc: SemanticDocument) -> None:
        self.documents[doc.document_id]=doc; self._matrix=None

    def _search_text(self, doc, p):
        aliases=" ".join(doc.entities.get(p.entity_id,{}).get("aliases",[])); s=p.scope
        return f"{aliases} {p.claim} temporal_scope={s.temporal_scope} epistemic_status={s.epistemic_status} proposition_polarity={s.proposition_polarity} speaker={s.speaker} condition={s.condition}"

    def build_index(self):
        self._records=[]; texts=[]
        for doc in self.documents.values():
            for p in doc.propositions:
                self._records.append((doc,p)); texts.append(self._search_text(doc,p))
        self._matrix=self.embedder.encode_passages(texts) if texts else np.empty((0,0))

    def search_units(self, query: str, *, top_k: int=50, entity_filter: bool=True) -> list[SemanticMatch]:
        """작은 semantic unit을 검색합니다. 일반 사용자는 search()를 권장합니다."""
        if self._matrix is None: self.build_index()
        if not self._records: return []
        scores=self._matrix @ self.embedder.encode_query(query)
        q_entities={eid for _,eid in entity_mentions(query)}; candidates=[]
        for idx,score in enumerate(scores.tolist()):
            doc,p=self._records[idx]
            if entity_filter and q_entities and p.entity_id not in q_entities: continue
            candidates.append(SemanticMatch(doc.document_id,p.id,p.entity_id,float(score),p))
        return sorted(candidates,key=lambda x:x.score,reverse=True)[:top_k]

    @staticmethod
    def _line_number(text: str, char_offset: int) -> int:
        return text.count("\n",0,max(0,char_offset))+1

    def _assemble_regions(self, anchors: list[SemanticMatch], *, top_k: int, before: int, after: int, max_context_chars: int) -> list[EvidenceRegion]:
        # 동일 문서에서 proposition index가 가까운 anchor들을 하나의 evidence region으로 병합합니다.
        # (Merge nearby anchors in the same document into one evidence region.)
        by_doc={}
        for a in anchors: by_doc.setdefault(a.document_id,[]).append(a)
        regions=[]
        for doc_id,items in by_doc.items():
            doc=self.documents[doc_id]
            prop_index={p.id:i for i,p in enumerate(doc.propositions)}
            selected=[]
            for a in sorted(items,key=lambda x:prop_index[x.proposition_id]):
                idx=prop_index[a.proposition_id]
                lo=max(0,idx-before); hi=min(len(doc.propositions)-1,idx+after)
                selected.append([lo,hi,[a]])
            merged=[]
            for lo,hi,group in selected:
                if merged and lo <= merged[-1][1]+1:
                    merged[-1][1]=max(merged[-1][1],hi); merged[-1][2].extend(group)
                else: merged.append([lo,hi,list(group)])
            for lo,hi,group in merged:
                props=doc.propositions[lo:hi+1]
                start=min(p.source.start for p in props); end=max(p.source.end for p in props)
                # char budget을 넘으면 anchor 중심으로 보수적으로 축소합니다.
                if end-start > max_context_chars:
                    anchor_start=min(a.proposition.source.start for a in group)
                    anchor_end=max(a.proposition.source.end for a in group)
                    pad=max(0,(max_context_chars-(anchor_end-anchor_start))//2)
                    start=max(0,anchor_start-pad); end=min(len(doc.text),start+max_context_chars)
                text=doc.text[start:end]
                regions.append(EvidenceRegion(
                    document_id=doc_id,
                    score=max(a.score for a in group),
                    start_char=start,end_char=end,
                    start_line=self._line_number(doc.text,start),
                    end_line=self._line_number(doc.text,end),
                    text=text,
                    anchor_proposition_ids=sorted({a.proposition_id for a in group},key=lambda pid:prop_index[pid]),
                    entity_ids=sorted({a.entity_id for a in group}),
                ))
        return sorted(regions,key=lambda r:r.score,reverse=True)[:top_k]

    def search(self, query: str, *, top_k: int=10, candidate_k: int|None=None, entity_filter: bool=True, before: int=2, after: int=2, max_context_chars: int=4000) -> list[EvidenceRegion]:
        """Top-K semantic anchors가 아니라 Top-K 동적 evidence regions를 반환합니다."""
        candidate_k=candidate_k or max(top_k*10,50)
        anchors=self.search_units(query,top_k=candidate_k,entity_filter=entity_filter)
        return self._assemble_regions(anchors,top_k=top_k,before=before,after=after,max_context_chars=max_context_chars)

    def build_context(self, results: list[EvidenceRegion]) -> str:
        lines=["=== EVIDENCE REGIONS ==="]
        for r in results:
            lines += [
                f"[{r.document_id}] lines={r.start_line}-{r.end_line} score={r.score:.4f}",
                f"anchors={','.join(r.anchor_proposition_ids)} entities={','.join(r.entity_ids)}",
                r.text,
                "",
            ]
        return "\n".join(lines).strip()

    def save(self,path: str):
        with open(path,"w",encoding="utf-8") as f: json.dump([d.to_dict() for d in self.documents.values()],f,ensure_ascii=False,indent=2)
