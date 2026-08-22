from __future__ import annotations
from dataclasses import dataclass, asdict, field
from typing import Any


@dataclass
class Scope:
    """독립적인 의미 범위 축 (Independent semantic scope dimensions)."""

    proposition_polarity: str = "affirmative"
    epistemic_status: str = "asserted"
    temporal_scope: str = ""
    condition: str = ""
    speaker: str = ""

    @property
    def polarity(self) -> str:
        return "positive" if self.proposition_polarity == "affirmative" else "negative"

    @property
    def stance(self) -> str:
        return self.epistemic_status

    @property
    def modality(self) -> str:
        return self.epistemic_status

    @property
    def time(self) -> str:
        return self.temporal_scope

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Scope":
        polarity = data.get("proposition_polarity")
        if polarity is None:
            polarity = "negative" if data.get("polarity") == "negative" else "affirmative"
        epistemic = data.get(
            "epistemic_status",
            data.get("stance", data.get("modality", "asserted")),
        )
        temporal = data.get("temporal_scope", data.get("time", ""))
        return cls(
            proposition_polarity=polarity,
            epistemic_status=epistemic,
            temporal_scope=temporal,
            condition=data.get("condition", ""),
            speaker=data.get("speaker", ""),
        )


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
    source_uri: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self, *, ensure_ascii: bool = False, indent: int = 2) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SemanticDocument":
        props = [
            Proposition(
                id=p["id"],
                entity_id=p["entity_id"],
                claim=p["claim"],
                source=SourceSpan(**p["source"]),
                scope=Scope.from_dict(p.get("scope", {})),
                importance=p.get("importance", "supporting"),
            )
            for p in data.get("propositions", [])
        ]
        rels = [Relation(**r) for r in data.get("relations", [])]
        return cls(
            document_id=data["document_id"],
            language=data.get("language", "auto"),
            text=data.get("text", ""),
            entities=data.get("entities", {}),
            propositions=props,
            relations=rels,
            diagnostics=data.get("diagnostics", {}),
            source_uri=data.get("source_uri", ""),
        )
