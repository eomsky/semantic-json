from __future__ import annotations
import re
from .schemas import Scope, SourceSpan, Proposition, Relation, SemanticDocument

KO_UNCERTAIN=("단정하기 어렵","불확실","확인 필요","확정하기 어렵","배제할 수 없")
EN_UNCERTAIN=("difficult to conclude","uncertain","not yet confirmed","cannot conclude","cannot rule out")
KO_POSSIBLE=("가능성이","가능할","수 있다","우려")
EN_POSSIBLE=("may ","might ","could ","possible","possibly")
KO_CONDITION=("경우","전제로","조건","유지된다면","가정")
EN_CONDITION=(" if ","provided that","assuming","subject to","unless")
KO_CONTRAST=("그러나","다만","반면","불구하고","그럼에도")
EN_CONTRAST=("however","although","but ","whereas","nevertheless","despite")
KO_CAUSE=("때문","따라","원인","영향으로")
EN_CAUSE=("because","due to","therefore","as a result","accordingly")
KO_FUTURE=("내년","향후","예정","전망")
EN_FUTURE=("next year","future","scheduled","expected to","forecast")
KO_NEG=("아니","없","않","못")
EN_NEG=(" not "," no ","never","without")

def detect_language(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"

def split_sentences(text: str):
    # 원문 offset을 보존하는 문장 분리 (Sentence splitting with source offsets)
    spans=[]; start=0
    for m in re.finditer(r"(?<=[.!?])\s+|\n{2,}", text):
        end=m.start(); s=text[start:end].strip()
        if s:
            rs=text.find(s,start,end+1); spans.append((rs,rs+len(s),s))
        start=m.end()
    s=text[start:].strip()
    if s:
        rs=text.find(s,start); spans.append((rs,rs+len(s),s))
    return spans

def normalize_entity(surface: str) -> str:
    m=re.fullmatch(r"([A-Za-z])\s*(?:기업|Corp\.?|Corporation)", surface.strip(), re.I)
    if m: return m.group(1).upper()+"_CORP"
    return re.sub(r"[^A-Za-z0-9가-힣]+","_",surface).strip("_").upper()

def entity_mentions(sentence: str):
    out=[]
    for pat in [r"\b([A-Z])\s+Corp\.?\b",r"\b([A-Z])\s+Corporation\b",r"(?<![A-Za-z])([A-Z])기업"]:
        for m in re.finditer(pat,sentence):
            surface=m.group(0); out.append((surface,normalize_entity(surface)))
    return out

def scope_for(sentence: str, language: str) -> Scope:
    t=" "+sentence.lower()+" "
    if language=="ko":
        uncertain=any(x in sentence for x in KO_UNCERTAIN); possible=any(x in sentence for x in KO_POSSIBLE); condition=next((x for x in KO_CONDITION if x in sentence),""); future=any(x in sentence for x in KO_FUTURE); negative=any(x in sentence for x in KO_NEG)
    else:
        uncertain=any(x in t for x in EN_UNCERTAIN); possible=any(x in t for x in EN_POSSIBLE); condition=next((x.strip() for x in EN_CONDITION if x in t),""); future=any(x in t for x in EN_FUTURE); negative=any(x in t for x in EN_NEG)
    if "단정하기 어렵" in sentence or "difficult to conclude" in t: stance="difficult_to_conclude"
    elif uncertain: stance="uncertain"
    elif possible: stance="possible"
    else: stance="asserted"
    speaker="company" if ("회사 측" in sentence or "management" in t or "company expects" in t) else ""
    if "당행" in sentence or "bank reviewer" in t or "the bank" in t: speaker="bank"
    return Scope(polarity="negative" if negative else "positive",modality=stance if stance!="asserted" else "asserted",stance=stance,time="future" if future else "",condition=condition,speaker=speaker)

def importance_for(sentence: str, scope: Scope) -> str:
    if scope.stance in {"uncertain","difficult_to_conclude"}: return "core"
    finance=("상환","차입","부채","유동성","현금","수익","매출","이익","담보","만기","repay","debt","borrow","liquidity","cash","profit","revenue","margin","maturity","leverage")
    return "supporting" if any(x in sentence.lower() for x in finance) else "background"

def relation_types(sentence: str, language: str):
    t=sentence.lower(); pairs=[]
    groups=[("contrast",KO_CONTRAST if language=="ko" else EN_CONTRAST),("causal_or_consequential",KO_CAUSE if language=="ko" else EN_CAUSE),("condition",KO_CONDITION if language=="ko" else EN_CONDITION)]
    for kind,markers in groups:
        for marker in markers:
            if marker.strip().lower() in t: pairs.append((kind,marker.strip()))
    return pairs

def compile(text: str, *, document_id: str="document", language: str="auto") -> SemanticDocument:
    """장문의 plain text를 SemanticDocument로 변환 (Compile long plain text into a SemanticDocument)."""
    lang=detect_language(text) if language=="auto" else language
    entities={}; propositions=[]; relations=[]; current_entity="UNKNOWN"
    for i,(start,end,sentence) in enumerate(split_sentences(text),1):
        mentions=entity_mentions(sentence)
        if mentions:
            current_entity=mentions[-1][1]
            for surface,eid in mentions:
                entities.setdefault(eid,{"id":eid,"aliases":[]})
                if surface not in entities[eid]["aliases"]: entities[eid]["aliases"].append(surface)
        sc=scope_for(sentence,lang); pid=f"P{i}"
        propositions.append(Proposition(id=pid,entity_id=current_entity,claim=sentence,source=SourceSpan(start=start,end=end,text=sentence),scope=sc,importance=importance_for(sentence,sc)))
        for rtype,marker in relation_types(sentence,lang): relations.append(Relation(id=f"R{len(relations)+1}",type=rtype,from_id=pid,marker=marker))
    return SemanticDocument(document_id=document_id,language=lang,text=text,entities=entities,propositions=propositions,relations=relations,diagnostics={"compiler":"semantic_json_rule_compiler_v0.1","source_grounded":all(p.source.text in text for p in propositions),"unknown_entity_propositions":sum(p.entity_id=="UNKNOWN" for p in propositions)})
