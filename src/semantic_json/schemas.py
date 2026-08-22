from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any

@dataclass
class Scope:
    polarity: str = "positive"
    modality: str = "asserted"
    stance: str = "asserted"
    time: str = ""
    condition: str = ""
    speaker: str = ""

@dataclass
class SourceSpan:
    start: int
    end: int
    text: str

@dataclass
class Proposition:
    id: str
    entity_id: str
    claim: str
    source: SourceSpan
    scope: Scope = field(default_factory=Scope)
    importance: str = "supporting"

@dataclass
class Relation:
    id: str
    type: str
    from_id: str
    to_id: str | None = None
    marker: str = ""

@dataclass
class SemanticDocument:
    document_id: str
    language: str
    text: str
    entities: dict[str, dict[str, Any]]
    propositions: list[Proposition]
    relations: list[Relation]
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, ensure_ascii: bool = False, indent: int = 2) -> str:
        import json
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticDocument":
        props = [Proposition(id=p["id"], entity_id=p["entity_id"], claim=p["claim"], source=SourceSpan(**p["source"]), scope=Scope(**p.get("scope", {})), importance=p.get("importance", "supporting")) for p in data.get("propositions", [])]
        rels = [Relation(**r) for r in data.get("relations", [])]
        return cls(document_id=data["document_id"], language=data.get("language", "auto"), text=data.get("text", ""), entities=data.get("entities", {}), propositions=props, relations=rels, diagnostics=data.get("diagnostics", {}))
