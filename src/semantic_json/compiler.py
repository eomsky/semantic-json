from __future__ import annotations
import re
from .schemas import Scope, SourceSpan, Proposition, Relation, SemanticDocument

KO_UNCERTAIN_PATTERNS=(r"단정(?:하기|하기는|하기가)?\s*어렵",r"확정(?:하기|하기는|하기가)?\s*어렵",r"불확실",r"확인(?:이)?\s*필요",r"배제할\s*수\s*없")
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
KO_MEDIUM_TERM=("중장기","중장기적","중기","장기적으로")
EN_FUTURE=("next year","future","scheduled","expected to","forecast")
EN_MEDIUM_TERM=("medium term","medium-term","long term","long-term")
KO_NEG=("아니","없","않","못")
EN_NEG=(" not "," no ","never","without")


def detect_language(text: str) -> str:
    return "ko" if re.search(r"[가-힣]", text) else "en"


def split_sentences(text: str):
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


def split_clauses(sentence: str, language: str):
    if language=="ko":
        pattern=r"(?<=며),?\s+(?=(?:다만\s+)?(?:중장기|중기|장기적으로|현재|향후|내년))"
    else:
        pattern=r"\s*;\s*|,?\s+(?=(?:however|nevertheless|accordingly|therefore)\b)"
    pieces=[]; last=0
    for m in re.finditer(pattern,sentence,flags=re.I):
        part=sentence[last:m.start()].strip(" ,")
        if part:
            offset=sentence.find(part,last,m.start()+1); pieces.append((offset,offset+len(part),part))
        last=m.end()
    part=sentence[last:].strip(" ,")
    if part:
        offset=sentence.find(part,last); pieces.append((offset,offset+len(part),part))
    return pieces or [(0,len(sentence),sentence)]


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


def scope_for(clause: str, language: str) -> Scope:
    t=" "+clause.lower()+" "
    if language=="ko":
        difficult=bool(re.search(r"단정(?:하기|하기는|하기가)?\s*어렵",clause))
        uncertain=difficult or any(re.search(p,clause) for p in KO_UNCERTAIN_PATTERNS)
        possible=any(x in clause for x in KO_POSSIBLE)
        condition=next((x for x in KO_CONDITION if x in clause),"")
        if any(x in clause for x in KO_MEDIUM_TERM): temporal="medium_term"
        elif any(x in clause for x in KO_FUTURE): temporal="future"
        elif "현재" in clause or "현재까지" in clause: temporal="current"
        else: temporal=""
        negative=any(x in clause for x in KO_NEG)
    else:
        difficult="difficult to conclude" in t
        uncertain=difficult or any(x in t for x in EN_UNCERTAIN)
        possible=any(x in t for x in EN_POSSIBLE)
        condition=next((x.strip() for x in EN_CONDITION if x in t),"")
        if any(x in t for x in EN_MEDIUM_TERM): temporal="medium_term"
        elif any(x in t for x in EN_FUTURE): temporal="future"
        elif " current" in t or "to date" in t: temporal="current"
        else: temporal=""
        negative=any(x in t for x in EN_NEG)
    if difficult: epistemic="difficult_to_conclude"
    elif uncertain: epistemic="uncertain"
    elif possible: epistemic="possible"
    else: epistemic="asserted"
    speaker="company" if ("회사 측" in clause or "management" in t or "company expects" in t) else ""
    if "당행" in clause or "bank reviewer" in t or "the bank" in t: speaker="bank"
    return Scope(
        proposition_polarity="negative" if negative else "affirmative",
        epistemic_status=epistemic,
        temporal_scope=temporal,
        condition=condition,
        speaker=speaker,
    )


def importance_for(clause: str, scope: Scope) -> str:
    if scope.epistemic_status in {"uncertain","difficult_to_conclude"}: return "core"
    finance=("상환","차입","부채","유동성","현금","수익","매출","이익","담보","만기","repay","debt","borrow","liquidity","cash","profit","revenue","margin","maturity","leverage")
    return "supporting" if any(x in clause.lower() for x in finance) else "background"


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
    entities={}; propositions=[]; relations=[]; current_entity="UNKNOWN"; pid_n=1
    for start,end,sentence in split_sentences(text):
        mentions=entity_mentions(sentence)
        if mentions:
            current_entity=mentions[-1][1]
            for surface,eid in mentions:
                entities.setdefault(eid,{"id":eid,"aliases":[]})
                if surface not in entities[eid]["aliases"]: entities[eid]["aliases"].append(surface)
        sentence_prop_ids=[]
        for local_start,local_end,clause in split_clauses(sentence,lang):
            clause_mentions=entity_mentions(clause)
            clause_entity=clause_mentions[-1][1] if clause_mentions else current_entity
            sc=scope_for(clause,lang); pid=f"P{pid_n}"; pid_n+=1
            abs_start=start+local_start; abs_end=start+local_end
            propositions.append(Proposition(id=pid,entity_id=clause_entity,claim=clause,source=SourceSpan(start=abs_start,end=abs_end,text=clause),scope=sc,importance=importance_for(clause,sc)))
            sentence_prop_ids.append(pid)
        for rtype,marker in relation_types(sentence,lang):
            from_id=sentence_prop_ids[0] if sentence_prop_ids else ""
            to_id=sentence_prop_ids[1] if len(sentence_prop_ids)>1 else None
            relations.append(Relation(id=f"R{len(relations)+1}",type=rtype,from_id=from_id,to_id=to_id,marker=marker))
    return SemanticDocument(document_id=document_id,language=lang,text=text,entities=entities,propositions=propositions,relations=relations,diagnostics={"compiler":"semantic_json_rule_compiler_v0.1.0a4","grammar":"semantic_json_grammar_v0.1","source_grounded":all(p.source.text in text for p in propositions),"unknown_entity_propositions":sum(p.entity_id=="UNKNOWN" for p in propositions)})
