from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import random
from typing import Callable, Iterable, Mapping, Sequence

import numpy as np

from .embeddings import LiteEmbedder
from .units import FineGrainedUnitizer, SemanticUnit


DATASET_SCHEMA = "semantic-json-transport/region-dataset/v0.1"
SAME_REGION = "same_region"
SPLIT = "split"


@dataclass(frozen=True)
class RegionQuery:
    query: str
    query_id: str = ""
    document_id: str = ""


@dataclass(frozen=True)
class RegionCandidate:
    query: str
    left: str
    right: str
    query_id: str = ""
    left_document_id: str = ""
    right_document_id: str = ""
    left_unit_id: str = ""
    right_unit_id: str = ""
    candidate_type: str = "adjacent"
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class TeacherDecision:
    decision: str
    score: float
    confidence: str = ""
    reason_codes: tuple[str, ...] = ()


@dataclass
class RegionTrainingExample:
    query: str
    left: str
    right: str
    label: float
    decision: str = ""
    teacher: str = ""
    teacher_score: float | None = None
    confidence: str = ""
    reason_codes: list[str] = field(default_factory=list)
    query_id: str = ""
    left_document_id: str = ""
    right_document_id: str = ""
    left_unit_id: str = ""
    right_unit_id: str = ""
    candidate_type: str = ""
    schema: str = DATASET_SCHEMA
    metadata: dict = field(default_factory=dict)


class LLMRegionTeacher:
    """Provider-neutral adapter around a user supplied LLM/judge callable.

    The callable receives ``(query, left, right)``. Supported returns include:
    ``bool``, numeric score, ``"same_region"`` / ``"split"``, or a dict such as::

        {
            "label": "same_region",
            "score": 0.91,
            "confidence": "high",
            "reason_codes": ["same_evidence_chain"]
        }

    Scores are treated as teacher scores, not calibrated probabilities.
    No LLM SDK is imported or required by this package.
    """

    def __init__(self, judge: Callable[[str, str, str], object], *, name: str = "user-llm"):
        self.judge = judge
        self.name = name

    @staticmethod
    def _decision_from_value(value: object) -> tuple[str, float]:
        if isinstance(value, bool):
            return (SAME_REGION if value else SPLIT, 1.0 if value else 0.0)
        if isinstance(value, (int, float)):
            score = max(0.0, min(1.0, float(value)))
            return (SAME_REGION if score >= 0.5 else SPLIT, score)
        if isinstance(value, str):
            normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in {SAME_REGION, "same", "join", "together", "1", "true"}:
                return SAME_REGION, 1.0
            if normalized in {SPLIT, "separate", "boundary", "0", "false"}:
                return SPLIT, 0.0
        raise TypeError("Unsupported teacher label. Use bool, score, same_region/split, or a dict.")

    def decide(self, query: str, left: str, right: str) -> TeacherDecision:
        result = self.judge(query, left, right)
        if isinstance(result, dict):
            raw_label = result.get("label", result.get("decision"))
            raw_score = result.get("score")
            if raw_label is None and raw_score is None:
                raise TypeError("Teacher dict must contain label/decision and/or score.")
            if raw_label is not None:
                decision, fallback_score = self._decision_from_value(raw_label)
            else:
                decision, fallback_score = self._decision_from_value(raw_score)
            score = fallback_score if raw_score is None else max(0.0, min(1.0, float(raw_score)))
            confidence = str(result.get("confidence", ""))
            reasons = result.get("reason_codes", result.get("reasons", [])) or []
            if isinstance(reasons, str):
                reasons = [reasons]
            return TeacherDecision(decision, score, confidence, tuple(str(x) for x in reasons))
        decision, score = self._decision_from_value(result)
        return TeacherDecision(decision, score)

    def label(self, query: str, left: str, right: str) -> float:
        """Backward-compatible numeric teacher-score API."""
        return self.decide(query, left, right).score


class RegionDatasetBuilder:
    """Generate and label Region Compatibility Dataset v0.1 examples.

    Candidate generation is deliberately separate from teacher labeling. Hard candidates
    are *not* assumed to be negatives: the teacher still decides SAME_REGION vs SPLIT.
    This makes it safe to mine semantically similar but non-adjacent/cross-document pairs.
    """

    def __init__(
        self,
        teacher: LLMRegionTeacher | None = None,
        *,
        unitizer=None,
        embedder=None,
        random_seed: int = 13,
    ):
        self.teacher = teacher
        self.unitizer = unitizer or FineGrainedUnitizer()
        self.embedder = embedder or LiteEmbedder()
        self.random = random.Random(random_seed)

    @staticmethod
    def _coerce_queries(queries: Iterable[str | RegionQuery]) -> list[RegionQuery]:
        rows = []
        for i, item in enumerate(queries):
            if isinstance(item, RegionQuery):
                rows.append(item)
            else:
                rows.append(RegionQuery(str(item), query_id=f"Q{i + 1}"))
        return rows

    def _unitize_documents(self, documents: Mapping[str, str]) -> dict[str, list[SemanticUnit]]:
        return {
            doc_id: self.unitizer.unitize(text, document_id=doc_id)
            for doc_id, text in documents.items()
        }

    @staticmethod
    def _adjacent_candidates(query: RegionQuery, units: Sequence[SemanticUnit]) -> list[RegionCandidate]:
        return [
            RegionCandidate(
                query=query.query,
                query_id=query.query_id,
                left=left.text,
                right=right.text,
                left_document_id=left.document_id,
                right_document_id=right.document_id,
                left_unit_id=left.id,
                right_unit_id=right.id,
                candidate_type="adjacent",
                metadata={
                    "same_paragraph": left.paragraph_id == right.paragraph_id,
                    "source_gap_chars": max(0, right.start_char - left.end_char),
                },
            )
            for left, right in zip(units, units[1:])
        ]

    def _hard_candidates(
        self,
        query: RegionQuery,
        units_by_doc: Mapping[str, Sequence[SemanticUnit]],
        *,
        limit: int,
    ) -> list[RegionCandidate]:
        if limit <= 0:
            return []
        pool = [u for units in units_by_doc.values() for u in units]
        if len(pool) < 2:
            return []
        matrix = self.embedder.encode_passages([u.text for u in pool])
        qvec = self.embedder.encode_query(query.query)
        q_scores = matrix @ qvec
        # Focus on query-relevant units, then pair semantically similar units that are
        # non-adjacent or from different documents. These are useful hard boundary cases.
        top_indices = np.argsort(q_scores)[::-1][: min(len(pool), max(12, limit * 4))]
        scored_pairs: list[tuple[float, int, int]] = []
        for pos, i in enumerate(top_indices):
            for j in top_indices[pos + 1 :]:
                left, right = pool[int(i)], pool[int(j)]
                same_doc = left.document_id == right.document_id
                if same_doc:
                    try:
                        li = int(left.id.lstrip("U"))
                        ri = int(right.id.lstrip("U"))
                        if abs(li - ri) <= 1:
                            continue
                    except ValueError:
                        pass
                pair_similarity = float(matrix[int(i)] @ matrix[int(j)])
                relevance = float((q_scores[int(i)] + q_scores[int(j)]) / 2)
                scored_pairs.append((0.6 * relevance + 0.4 * pair_similarity, int(i), int(j)))
        scored_pairs.sort(reverse=True, key=lambda x: x[0])
        rows = []
        for mined_score, i, j in scored_pairs[:limit]:
            left, right = pool[i], pool[j]
            rows.append(
                RegionCandidate(
                    query=query.query,
                    query_id=query.query_id,
                    left=left.text,
                    right=right.text,
                    left_document_id=left.document_id,
                    right_document_id=right.document_id,
                    left_unit_id=left.id,
                    right_unit_id=right.id,
                    candidate_type="hard_candidate",
                    metadata={"mining_score": mined_score},
                )
            )
        return rows

    def _easy_candidates(
        self,
        query: RegionQuery,
        units_by_doc: Mapping[str, Sequence[SemanticUnit]],
        *,
        limit: int,
    ) -> list[RegionCandidate]:
        if limit <= 0:
            return []
        pool = [u for units in units_by_doc.values() for u in units]
        if len(pool) < 2:
            return []
        pairs = []
        attempts = 0
        while len(pairs) < limit and attempts < limit * 30:
            attempts += 1
            left, right = self.random.sample(pool, 2)
            if left.document_id == right.document_id:
                try:
                    if abs(int(left.id.lstrip("U")) - int(right.id.lstrip("U"))) <= 1:
                        continue
                except ValueError:
                    pass
            key = (left.document_id, left.id, right.document_id, right.id)
            if any(x.metadata.get("pair_key") == key for x in pairs):
                continue
            pairs.append(
                RegionCandidate(
                    query=query.query,
                    query_id=query.query_id,
                    left=left.text,
                    right=right.text,
                    left_document_id=left.document_id,
                    right_document_id=right.document_id,
                    left_unit_id=left.id,
                    right_unit_id=right.id,
                    candidate_type="easy_candidate",
                    metadata={"pair_key": key},
                )
            )
        return pairs

    def generate_candidates(
        self,
        *,
        documents: Mapping[str, str],
        queries: Iterable[str | RegionQuery],
        adjacent_per_query: int | None = 100,
        hard_candidates_per_query: int = 20,
        easy_candidates_per_query: int = 10,
    ) -> list[RegionCandidate]:
        units_by_doc = self._unitize_documents(documents)
        rows: list[RegionCandidate] = []
        for query in self._coerce_queries(queries):
            target_docs = (
                {query.document_id: units_by_doc.get(query.document_id, [])}
                if query.document_id
                else units_by_doc
            )
            adjacent = []
            for units in target_docs.values():
                adjacent.extend(self._adjacent_candidates(query, units))
            if adjacent_per_query is not None and len(adjacent) > adjacent_per_query:
                # Preserve source diversity without making candidate generation unbounded.
                adjacent = self.random.sample(adjacent, adjacent_per_query)
            rows.extend(adjacent)
            rows.extend(self._hard_candidates(query, units_by_doc, limit=hard_candidates_per_query))
            rows.extend(self._easy_candidates(query, units_by_doc, limit=easy_candidates_per_query))
        return rows

    def label_candidates(
        self,
        candidates: Iterable[RegionCandidate],
        *,
        teacher: LLMRegionTeacher | None = None,
        include_low_confidence: bool = True,
    ) -> list[RegionTrainingExample]:
        teacher = teacher or self.teacher
        if teacher is None:
            raise ValueError("A teacher is required to label candidates.")
        rows = []
        for candidate in candidates:
            decision = teacher.decide(candidate.query, candidate.left, candidate.right)
            if not include_low_confidence and decision.confidence.lower() in {"low", "uncertain"}:
                continue
            rows.append(
                RegionTrainingExample(
                    query=candidate.query,
                    left=candidate.left,
                    right=candidate.right,
                    label=decision.score,
                    decision=decision.decision,
                    teacher=teacher.name,
                    teacher_score=decision.score,
                    confidence=decision.confidence,
                    reason_codes=list(decision.reason_codes),
                    query_id=candidate.query_id,
                    left_document_id=candidate.left_document_id,
                    right_document_id=candidate.right_document_id,
                    left_unit_id=candidate.left_unit_id,
                    right_unit_id=candidate.right_unit_id,
                    candidate_type=candidate.candidate_type,
                    metadata=dict(candidate.metadata),
                )
            )
        return rows

    def build(self, *, documents: Mapping[str, str], queries: Iterable[str | RegionQuery], **kwargs) -> list[RegionTrainingExample]:
        candidates = self.generate_candidates(documents=documents, queries=queries, **kwargs)
        return self.label_candidates(candidates)

    def label_pairs(self, pairs: Iterable[tuple[str, str, str]]) -> list[RegionTrainingExample]:
        """Backward-compatible direct pair-labeling API."""
        candidates = [RegionCandidate(query=q, left=l, right=r, candidate_type="custom") for q, l, r in pairs]
        return self.label_candidates(candidates)

    @staticmethod
    def save_jsonl(examples: Iterable[RegionTrainingExample], path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")

    @staticmethod
    def load_jsonl(path: str | Path, *, validate_schema: bool = True) -> list[RegionTrainingExample]:
        rows = []
        with open(path, encoding="utf-8") as f:
            for line_number, line in enumerate(f, 1):
                if not line.strip():
                    continue
                payload = json.loads(line)
                if validate_schema and payload.get("schema", DATASET_SCHEMA) != DATASET_SCHEMA:
                    raise ValueError(f"Unsupported dataset schema at line {line_number}: {payload.get('schema')}")
                payload.setdefault("schema", DATASET_SCHEMA)
                rows.append(RegionTrainingExample(**payload))
        return rows

    @staticmethod
    def summarize(examples: Iterable[RegionTrainingExample]) -> dict:
        rows = list(examples)
        counts = {SAME_REGION: 0, SPLIT: 0}
        types: dict[str, int] = {}
        confidence: dict[str, int] = {}
        for row in rows:
            decision = row.decision or (SAME_REGION if row.label >= 0.5 else SPLIT)
            counts[decision] = counts.get(decision, 0) + 1
            types[row.candidate_type] = types.get(row.candidate_type, 0) + 1
            key = row.confidence or "unspecified"
            confidence[key] = confidence.get(key, 0) + 1
        return {
            "schema": DATASET_SCHEMA,
            "examples": len(rows),
            "decisions": counts,
            "candidate_types": types,
            "confidence": confidence,
        }


class RegionEncoderTrainer:
    """Explicit opt-in fine-tuning of a cross-encoder on Region Dataset examples."""

    def __init__(self, base_model: str, *, device: str = "cpu"):
        self.base_model = base_model
        self.device = device

    def fit(
        self,
        examples: list[RegionTrainingExample],
        *,
        output_path: str,
        epochs: int = 1,
        batch_size: int = 16,
        warmup_steps: int = 0,
        min_teacher_confidence: str | None = None,
    ) -> str:
        if not examples:
            raise ValueError("At least one training example is required.")
        try:
            from torch.utils.data import DataLoader
            from sentence_transformers import CrossEncoder, InputExample
        except ImportError as exc:
            raise ImportError(
                'Fine-tuning requires training dependencies. Install with: '
                'pip install "semantic-json-transport[training]"'
            ) from exc

        allowed = {"high": 3, "medium": 2, "low": 1, "": 0, "unspecified": 0}
        minimum = allowed.get((min_teacher_confidence or "").lower(), 0)
        selected = [
            row for row in examples
            if allowed.get((row.confidence or "").lower(), 0) >= minimum
        ]
        if not selected:
            raise ValueError("No training examples remain after confidence filtering.")
        train_rows = [
            InputExample(
                texts=[row.query, f"[LEFT]\n{row.left}\n[RIGHT]\n{row.right}"],
                label=float(row.label),
            )
            for row in selected
        ]
        loader = DataLoader(train_rows, shuffle=True, batch_size=batch_size)
        model = CrossEncoder(self.base_model, num_labels=1, device=self.device)
        model.fit(
            train_dataloader=loader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            output_path=output_path,
            show_progress_bar=True,
        )
        return output_path
