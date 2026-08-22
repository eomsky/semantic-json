from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Protocol


DEFAULT_REGION_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"


class RegionCompatibilityModel(Protocol):
    """Score whether two adjacent source spans should travel in one region for a query."""

    def score(self, query: str, left: str, right: str) -> float:
        ...


@dataclass
class HeuristicRegionModel:
    """Dependency-free fallback. Scores are compatibility scores, not probabilities."""

    same_paragraph_bonus: float = 0.1

    @staticmethod
    def _tokens(text: str) -> set[str]:
        import re

        return set(re.findall(r"[A-Za-z0-9가-힣]+", text.lower()))

    def score(self, query: str, left: str, right: str) -> float:
        q = self._tokens(query)
        l = self._tokens(left)
        r = self._tokens(right)
        if not l or not r:
            return 0.0
        pair_overlap = len(l & r) / max(1, len(l | r))
        ql = len(q & l) / max(1, len(q))
        qr = len(q & r) / max(1, len(q))
        # Preserve bridging context: one side may be weakly query-relevant if it is
        # coherent with the other side.
        value = 0.45 * pair_overlap + 0.275 * ql + 0.275 * qr
        return max(0.0, min(1.0, value))


class RegionCompatibilityEncoder:
    """CPU-capable cross-encoder composer with lazy model loading.

    The public task is region compatibility, not generic relevance. The current
    default checkpoint is a multilingual cross-encoder bootstrap checkpoint. It is
    intentionally replaceable with a package-trained or user-fine-tuned checkpoint.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_REGION_MODEL,
        *,
        device: str = "cpu",
        max_length: int = 512,
    ):
        self.model_name = model_name
        self.device = device
        self.max_length = max_length
        self._model = None

    def _load(self):
        if self._model is not None:
            return self._model
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError(
                "RegionCompatibilityEncoder requires sentence-transformers. "
                "Install semantic-json-transport with its default dependencies."
            ) from exc
        self._model = CrossEncoder(
            self.model_name,
            device=self.device,
            max_length=self.max_length,
        )
        return self._model

    @staticmethod
    def _pair_text(left: str, right: str) -> str:
        return f"[LEFT]\n{left}\n[RIGHT]\n{right}"

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            z = math.exp(-value)
            return 1.0 / (1.0 + z)
        z = math.exp(value)
        return z / (1.0 + z)

    def score(self, query: str, left: str, right: str) -> float:
        model = self._load()
        raw = model.predict([(query, self._pair_text(left, right))], show_progress_bar=False)
        value = float(raw[0])
        # Existing reranker checkpoints generally emit logits. A user-fine-tuned
        # region model can use the same interface; calibration is deliberately kept
        # separate from this raw compatibility score.
        return self._sigmoid(value)
