from __future__ import annotations

from dataclasses import dataclass
import json
import numpy as np

from .schemas import SemanticDocument, Proposition
from .embeddings import LiteEmbedder
from .compiler import entity_mentions


@dataclass
class SemanticMatch:
    """Internal semantic anchor returned by unit-level retrieval."""

    document_id: str
    proposition_id: str
    entity_id: str
    score: float
    proposition: Proposition


@dataclass
class EvidenceRegion:
    """Source-grounded context assembled dynamically at query time."""

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
    """Retrieve semantic anchors and assemble entity-safe source evidence regions."""

    def __init__(self, *, embedder=None):
        self.documents: dict[str, SemanticDocument] = {}
        self.embedder = embedder or LiteEmbedder()
        self._records: list[tuple[SemanticDocument, Proposition]] = []
        self._matrix = None

    def add(self, doc: SemanticDocument) -> None:
        self.documents[doc.document_id] = doc
        self._matrix = None

    def _search_text(self, doc: SemanticDocument, p: Proposition) -> str:
        aliases = " ".join(doc.entities.get(p.entity_id, {}).get("aliases", []))
        s = p.scope
        return (
            f"{aliases} {p.claim} temporal_scope={s.temporal_scope} "
            f"epistemic_status={s.epistemic_status} "
            f"proposition_polarity={s.proposition_polarity} "
            f"speaker={s.speaker} condition={s.condition}"
        )

    def build_index(self) -> None:
        self._records = []
        texts = []
        for doc in self.documents.values():
            for p in doc.propositions:
                self._records.append((doc, p))
                texts.append(self._search_text(doc, p))
        self._matrix = (
            self.embedder.encode_passages(texts)
            if texts
            else np.empty((0, 0), dtype=np.float32)
        )

    def search_units(
        self,
        query: str,
        *,
        top_k: int = 50,
        entity_filter: bool = True,
    ) -> list[SemanticMatch]:
        """Search small semantic units. Prefer search() for downstream LLM context."""
        if self._matrix is None:
            self.build_index()
        if not self._records:
            return []

        scores = self._matrix @ self.embedder.encode_query(query)
        q_entities = {eid for _, eid in entity_mentions(query)}
        candidates = []
        for idx, score in enumerate(scores.tolist()):
            doc, p = self._records[idx]
            if entity_filter and q_entities and p.entity_id not in q_entities:
                continue
            candidates.append(
                SemanticMatch(doc.document_id, p.id, p.entity_id, float(score), p)
            )
        return sorted(candidates, key=lambda x: x.score, reverse=True)[:top_k]

    @staticmethod
    def _line_number(text: str, char_offset: int) -> int:
        return text.count("\n", 0, max(0, char_offset)) + 1

    @staticmethod
    def _compatible_entity(candidate: Proposition, anchor_entities: set[str]) -> bool:
        """Prevent expansion across an explicit, unrelated entity boundary."""
        if candidate.entity_id == "UNKNOWN":
            return True
        return candidate.entity_id in anchor_entities

    @staticmethod
    def _fit_proposition_budget(
        props: list[Proposition],
        anchor_ids: set[str],
        max_context_chars: int,
    ) -> list[Proposition]:
        """Shrink only at proposition boundaries while preserving every anchor."""
        if not props:
            return []
        if max_context_chars <= 0:
            return props

        def span_len(items: list[Proposition]) -> int:
            return max(p.source.end for p in items) - min(p.source.start for p in items)

        if span_len(props) <= max_context_chars:
            return props

        anchor_positions = [i for i, p in enumerate(props) if p.id in anchor_ids]
        if not anchor_positions:
            return props

        lo, hi = min(anchor_positions), max(anchor_positions)
        kept = props[lo : hi + 1]

        # Anchors are never truncated, even when their exact source span exceeds the budget.
        if span_len(kept) >= max_context_chars:
            return kept

        left, right = lo - 1, hi + 1
        while left >= 0 or right < len(props):
            options = []
            if left >= 0:
                options.append((left, props[left : hi + 1]))
            if right < len(props):
                options.append((right, props[lo : right + 1]))

            added = False
            for idx, candidate in sorted(options, key=lambda x: abs(x[0] - (lo + hi) / 2)):
                if span_len(candidate) <= max_context_chars:
                    if idx < lo:
                        lo = idx
                        left = lo - 1
                    else:
                        hi = idx
                        right = hi + 1
                    kept = props[lo : hi + 1]
                    added = True
                    break
            if not added:
                break
        return kept

    def _expand_anchor_window(
        self,
        doc: SemanticDocument,
        anchor_index: int,
        anchor_entity: str,
        *,
        before: int,
        after: int,
        entity_safe: bool,
    ) -> tuple[int, int]:
        """Expand around an anchor without crossing an unrelated explicit entity."""
        props = doc.propositions
        anchor_entities = {anchor_entity}
        lo = hi = anchor_index

        for _ in range(before):
            nxt = lo - 1
            if nxt < 0:
                break
            if entity_safe and not self._compatible_entity(props[nxt], anchor_entities):
                break
            lo = nxt

        for _ in range(after):
            nxt = hi + 1
            if nxt >= len(props):
                break
            if entity_safe and not self._compatible_entity(props[nxt], anchor_entities):
                break
            hi = nxt

        return lo, hi

    def _assemble_regions(
        self,
        anchors: list[SemanticMatch],
        *,
        top_k: int,
        before: int,
        after: int,
        max_context_chars: int,
        entity_safe: bool,
    ) -> list[EvidenceRegion]:
        by_doc: dict[str, list[SemanticMatch]] = {}
        for anchor in anchors:
            by_doc.setdefault(anchor.document_id, []).append(anchor)

        regions = []
        for doc_id, items in by_doc.items():
            doc = self.documents[doc_id]
            prop_index = {p.id: i for i, p in enumerate(doc.propositions)}
            selected = []

            for anchor in sorted(items, key=lambda x: prop_index[x.proposition_id]):
                idx = prop_index[anchor.proposition_id]
                lo, hi = self._expand_anchor_window(
                    doc,
                    idx,
                    anchor.entity_id,
                    before=before,
                    after=after,
                    entity_safe=entity_safe,
                )
                selected.append([lo, hi, [anchor]])

            merged = []
            for lo, hi, group in selected:
                if merged and lo <= merged[-1][1] + 1:
                    existing_entities = {a.entity_id for a in merged[-1][2]}
                    incoming_entities = {a.entity_id for a in group}
                    can_merge = not entity_safe or bool(existing_entities & incoming_entities)
                    if can_merge:
                        merged[-1][1] = max(merged[-1][1], hi)
                        merged[-1][2].extend(group)
                        continue
                merged.append([lo, hi, list(group)])

            for lo, hi, group in merged:
                props = doc.propositions[lo : hi + 1]
                anchor_ids = {a.proposition_id for a in group}
                props = self._fit_proposition_budget(
                    props, anchor_ids, max_context_chars
                )
                if not props:
                    continue

                start = min(p.source.start for p in props)
                end = max(p.source.end for p in props)
                region_entity_ids = sorted(
                    {p.entity_id for p in props if p.entity_id != "UNKNOWN"}
                )
                regions.append(
                    EvidenceRegion(
                        document_id=doc_id,
                        score=max(a.score for a in group),
                        start_char=start,
                        end_char=end,
                        start_line=self._line_number(doc.text, start),
                        end_line=self._line_number(doc.text, max(start, end - 1)),
                        text=doc.text[start:end],
                        anchor_proposition_ids=sorted(
                            anchor_ids, key=lambda pid: prop_index[pid]
                        ),
                        entity_ids=region_entity_ids,
                    )
                )

        return sorted(regions, key=lambda r: r.score, reverse=True)[:top_k]

    def search(
        self,
        query: str,
        *,
        top_k: int = 10,
        candidate_k: int | None = None,
        entity_filter: bool = True,
        before: int = 2,
        after: int = 2,
        max_context_chars: int = 4000,
        entity_safe: bool = True,
    ) -> list[EvidenceRegion]:
        """Return Top-K source-grounded dynamic evidence regions."""
        candidate_k = candidate_k or max(top_k * 10, 50)
        anchors = self.search_units(
            query, top_k=candidate_k, entity_filter=entity_filter
        )
        return self._assemble_regions(
            anchors,
            top_k=top_k,
            before=before,
            after=after,
            max_context_chars=max_context_chars,
            entity_safe=entity_safe,
        )

    def build_context(self, results: list[EvidenceRegion]) -> str:
        lines = ["=== EVIDENCE REGIONS ==="]
        for r in results:
            lines += [
                f"[{r.document_id}] lines={r.start_line}-{r.end_line} score={r.score:.4f}",
                f"anchors={','.join(r.anchor_proposition_ids)} entities={','.join(r.entity_ids)}",
                r.text,
                "",
            ]
        return "\n".join(lines).strip()

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                [d.to_dict() for d in self.documents.values()],
                f,
                ensure_ascii=False,
                indent=2,
            )

    @classmethod
    def load(cls, path: str, *, embedder=None) -> "SemanticRepository":
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        repo = cls(embedder=embedder)
        for item in data:
            repo.add(SemanticDocument.from_dict(item))
        return repo
