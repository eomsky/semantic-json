from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from typing import Iterator

import numpy as np

from .embeddings import LiteEmbedder
from .region_models import HeuristicRegionModel, RegionCompatibilityEncoder, RegionCompatibilityModel
from .schemas import SemanticDocument
from .units import FineGrainedUnitizer, SemanticUnit


TRANSPORT_SCHEMA = "semantic-json-transport/context/v2"


@dataclass
class SourceDocument:
    document_id: str
    text: str
    source_uri: str = ""
    units: list[SemanticUnit] = field(default_factory=list)


@dataclass
class UnitMatch:
    document_id: str
    unit_id: str
    score: float
    unit: SemanticUnit


# Backward-compatible name from v0.1.
SemanticMatch = UnitMatch


@dataclass
class BoundaryDecision:
    left_unit_id: str
    right_unit_id: str
    compatibility: float
    included: bool

    def to_dict(self) -> dict:
        return {
            "left_unit_id": self.left_unit_id,
            "right_unit_id": self.right_unit_id,
            "compatibility_score": self.compatibility,
            "included": self.included,
        }


@dataclass
class EvidenceRegion:
    document_id: str
    score: float
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    text: str
    anchor_unit_ids: list[str]
    unit_ids: list[str]
    boundary_decisions: list[BoundaryDecision] = field(default_factory=list)
    region_id: str = ""
    source_uri: str = ""
    document_sha256: str = ""

    @property
    def anchor_proposition_ids(self) -> list[str]:
        return self.anchor_unit_ids

    @property
    def entity_ids(self) -> list[str]:
        return []

    def to_dict(self) -> dict:
        return {
            "region_id": self.region_id,
            "score": self.score,
            "source": {
                "document_id": self.document_id,
                "uri": self.source_uri,
                "start_char": self.start_char,
                "end_char": self.end_char,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "document_sha256": self.document_sha256,
            },
            "anchors": list(self.anchor_unit_ids),
            "units": list(self.unit_ids),
            "boundaries": [x.to_dict() for x in self.boundary_decisions],
            "text": self.text,
        }


@dataclass
class SearchResult:
    query: str
    regions: list[EvidenceRegion]
    schema: str = TRANSPORT_SCHEMA

    def __iter__(self) -> Iterator[EvidenceRegion]: return iter(self.regions)
    def __len__(self) -> int: return len(self.regions)
    def __getitem__(self, index): return self.regions[index]
    def __bool__(self) -> bool: return bool(self.regions)

    def to_dict(self) -> dict:
        documents: dict[str, dict] = {}
        for region in self.regions:
            doc = documents.setdefault(region.document_id, {
                "document_id": region.document_id,
                "source_uri": region.source_uri,
                "document_sha256": region.document_sha256,
                "regions": [],
            })
            doc["regions"].append(region.to_dict())
        return {
            "schema": self.schema,
            "query": self.query,
            "region_count": len(self.regions),
            "document_count": len(documents),
            "documents": list(documents.values()),
        }

    def to_json(self, *, ensure_ascii: bool = False, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=ensure_ascii, indent=indent)

    def to_text(self) -> str:
        lines = ["=== EVIDENCE REGIONS ===", f"query={self.query}"]
        for r in self.regions:
            source = f" source={r.source_uri}" if r.source_uri else ""
            lines += [
                "",
                f"[{r.region_id}] {r.document_id} lines={r.start_line}-{r.end_line} "
                f"chars={r.start_char}-{r.end_char} score={r.score:.4f}{source}",
                f"anchors={','.join(r.anchor_unit_ids)} units={','.join(r.unit_ids)}",
                r.text,
            ]
        return "\n".join(lines).strip()


class SemanticRepository:
    """Locator + query-conditioned region composer over original source text."""

    def __init__(
        self,
        *,
        embedder=None,
        region_model: RegionCompatibilityModel | str | None = None,
        unitizer=None,
        compatibility_threshold: float = 0.55,
    ):
        self.documents: dict[str, SourceDocument] = {}
        self.embedder = embedder or LiteEmbedder()
        if region_model == "lite":
            self.region_model = HeuristicRegionModel()
        elif region_model is None:
            self.region_model = RegionCompatibilityEncoder()
        else:
            self.region_model = region_model
        self.unitizer = unitizer or FineGrainedUnitizer()
        self.compatibility_threshold = compatibility_threshold
        self._records: list[tuple[SourceDocument, SemanticUnit]] = []
        self._matrix = None

    @staticmethod
    def _document_sha256(doc: SourceDocument) -> str:
        return hashlib.sha256(doc.text.encode("utf-8")).hexdigest()

    def add_text(self, text: str, *, document_id: str, source_uri: str = "") -> SourceDocument:
        doc = SourceDocument(
            document_id=document_id,
            text=text,
            source_uri=source_uri,
            units=self.unitizer.unitize(text, document_id=document_id),
        )
        self.documents[document_id] = doc
        self._matrix = None
        return doc

    def add(self, doc: SemanticDocument | SourceDocument) -> None:
        if isinstance(doc, SourceDocument):
            self.documents[doc.document_id] = doc
        else:
            # v0.1 compatibility: preserve existing proposition spans as source units.
            units = [SemanticUnit(
                id=f"U{i + 1}", document_id=doc.document_id,
                start_char=p.source.start, end_char=p.source.end,
                start_line=doc.text.count("\n", 0, p.source.start) + 1,
                end_line=doc.text.count("\n", 0, max(p.source.start, p.source.end - 1)) + 1,
                text=doc.text[p.source.start:p.source.end],
            ) for i, p in enumerate(doc.propositions)]
            self.documents[doc.document_id] = SourceDocument(
                doc.document_id, doc.text, getattr(doc, "source_uri", ""), units
            )
        self._matrix = None

    def build_index(self) -> None:
        self._records = []
        texts = []
        for doc in self.documents.values():
            for unit in doc.units:
                self._records.append((doc, unit))
                texts.append(unit.text)
        self._matrix = self.embedder.encode_passages(texts) if texts else np.empty((0, 0), dtype=np.float32)

    def search_units(self, query: str, *, top_k: int = 50, **_) -> list[UnitMatch]:
        if self._matrix is None: self.build_index()
        if not self._records: return []
        scores = self._matrix @ self.embedder.encode_query(query)
        matches = [UnitMatch(doc.document_id, unit.id, float(scores[i]), unit)
                   for i, (doc, unit) in enumerate(self._records)]
        return sorted(matches, key=lambda x: x.score, reverse=True)[:top_k]

    def _expand(self, query: str, doc: SourceDocument, anchor: UnitMatch,
                *, max_units: int, threshold: float) -> tuple[int, int, list[BoundaryDecision]]:
        index = {u.id: i for i, u in enumerate(doc.units)}
        lo = hi = index[anchor.unit_id]
        decisions: list[BoundaryDecision] = []
        while (hi - lo + 1) < max_units:
            candidates = []
            current_left = doc.text[doc.units[lo].start_char:doc.units[hi].end_char]
            if lo > 0:
                s = float(self.region_model.score(query, doc.units[lo - 1].text, current_left))
                candidates.append((s, "left"))
            if hi + 1 < len(doc.units):
                s = float(self.region_model.score(query, current_left, doc.units[hi + 1].text))
                candidates.append((s, "right"))
            if not candidates: break
            score, side = max(candidates, key=lambda x: x[0])
            if side == "left":
                left, right = doc.units[lo - 1], doc.units[lo]
            else:
                left, right = doc.units[hi], doc.units[hi + 1]
            included = score >= threshold
            decisions.append(BoundaryDecision(left.id, right.id, score, included))
            if not included: break
            if side == "left": lo -= 1
            else: hi += 1
        return lo, hi, decisions

    def _region_from_span(self, doc: SourceDocument, lo: int, hi: int,
                          anchors: list[UnitMatch], decisions: list[BoundaryDecision]) -> EvidenceRegion:
        units = doc.units[lo:hi + 1]
        start, end = units[0].start_char, units[-1].end_char
        return EvidenceRegion(
            document_id=doc.document_id,
            score=max(a.score for a in anchors),
            start_char=start, end_char=end,
            start_line=units[0].start_line, end_line=units[-1].end_line,
            text=doc.text[start:end],
            anchor_unit_ids=[a.unit_id for a in anchors],
            unit_ids=[u.id for u in units],
            boundary_decisions=decisions,
            source_uri=doc.source_uri,
            document_sha256=self._document_sha256(doc),
        )

    def search(self, query: str, *, top_k: int = 10, candidate_k: int | None = None,
               compatibility_threshold: float | None = None, max_region_units: int = 12,
               **_) -> SearchResult:
        candidate_k = candidate_k or max(top_k * 5, 20)
        threshold = self.compatibility_threshold if compatibility_threshold is None else compatibility_threshold
        anchors = self.search_units(query, top_k=candidate_k)
        candidates: list[tuple[str, int, int, UnitMatch, list[BoundaryDecision]]] = []
        for anchor in anchors:
            doc = self.documents[anchor.document_id]
            lo, hi, decisions = self._expand(query, doc, anchor, max_units=max_region_units, threshold=threshold)
            candidates.append((anchor.document_id, lo, hi, anchor, decisions))

        merged: list[EvidenceRegion] = []
        by_doc: dict[str, list[tuple[int, int, UnitMatch, list[BoundaryDecision]]]] = {}
        for doc_id, lo, hi, anchor, decisions in candidates:
            by_doc.setdefault(doc_id, []).append((lo, hi, anchor, decisions))
        for doc_id, spans in by_doc.items():
            doc = self.documents[doc_id]
            spans.sort(key=lambda x: x[0])
            groups: list[list] = []
            for lo, hi, anchor, decisions in spans:
                if groups and lo <= groups[-1][1] + 1:
                    groups[-1][1] = max(groups[-1][1], hi)
                    groups[-1][2].append(anchor)
                    groups[-1][3].extend(decisions)
                else:
                    groups.append([lo, hi, [anchor], list(decisions)])
            for lo, hi, group_anchors, decisions in groups:
                merged.append(self._region_from_span(doc, lo, hi, group_anchors, decisions))

        merged = sorted(merged, key=lambda x: x.score, reverse=True)[:top_k]
        counters: dict[str, int] = {}
        for region in merged:
            counters[region.document_id] = counters.get(region.document_id, 0) + 1
            region.region_id = f"{region.document_id}:R{counters[region.document_id]}"
        return SearchResult(query=query, regions=merged)

    def locate(self, region: EvidenceRegion) -> dict:
        return region.to_dict()["source"] | {"region_id": region.region_id}

    def get_source(self, region: EvidenceRegion, *, context_before: int = 0, context_after: int = 0) -> str:
        doc = self.documents[region.document_id]
        start = max(0, region.start_char - max(0, context_before))
        end = min(len(doc.text), region.end_char + max(0, context_after))
        return doc.text[start:end]

    def verify_source(self, region: EvidenceRegion) -> bool:
        doc = self.documents.get(region.document_id)
        return bool(doc and region.document_sha256 == self._document_sha256(doc)
                    and doc.text[region.start_char:region.end_char] == region.text)

    def build_context(self, results: SearchResult | list[EvidenceRegion]) -> str:
        return results.to_text() if isinstance(results, SearchResult) else SearchResult("", list(results)).to_text()

    def save(self, path: str) -> None:
        payload = [{"document_id": d.document_id, "text": d.text, "source_uri": d.source_uri}
                   for d in self.documents.values()]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str, **kwargs) -> "SemanticRepository":
        with open(path, encoding="utf-8") as f: data = json.load(f)
        repo = cls(**kwargs)
        for item in data:
            repo.add_text(item["text"], document_id=item["document_id"], source_uri=item.get("source_uri", ""))
        return repo
