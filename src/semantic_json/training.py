from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Callable, Iterable


@dataclass
class RegionTrainingExample:
    query: str
    left: str
    right: str
    label: float
    teacher: str = ""
    metadata: dict | None = None


class LLMRegionTeacher:
    """Adapter around a user-provided judge callable.

    The callable receives (query, left, right) and must return either a bool,
    a numeric score in [0, 1], or a dict containing `label`/`score`.
    No LLM provider is a runtime dependency of Semantic JSON Transport.
    """

    def __init__(self, judge: Callable[[str, str, str], object], *, name: str = "user-llm"):
        self.judge = judge
        self.name = name

    def label(self, query: str, left: str, right: str) -> float:
        result = self.judge(query, left, right)
        if isinstance(result, bool):
            return 1.0 if result else 0.0
        if isinstance(result, (int, float)):
            return max(0.0, min(1.0, float(result)))
        if isinstance(result, dict):
            value = result.get("score", result.get("label"))
            if isinstance(value, bool):
                return 1.0 if value else 0.0
            if isinstance(value, (int, float)):
                return max(0.0, min(1.0, float(value)))
        raise TypeError("Teacher judge must return bool, float, or {'label'/'score': value}.")


class RegionDatasetBuilder:
    def __init__(self, teacher: LLMRegionTeacher):
        self.teacher = teacher

    def label_pairs(
        self,
        pairs: Iterable[tuple[str, str, str]],
    ) -> list[RegionTrainingExample]:
        rows = []
        for query, left, right in pairs:
            rows.append(
                RegionTrainingExample(
                    query=query,
                    left=left,
                    right=right,
                    label=self.teacher.label(query, left, right),
                    teacher=self.teacher.name,
                )
            )
        return rows

    @staticmethod
    def save_jsonl(examples: Iterable[RegionTrainingExample], path: str | Path) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for example in examples:
                f.write(json.dumps(asdict(example), ensure_ascii=False) + "\n")


class RegionEncoderTrainer:
    """Fine-tune a cross-encoder on SAME_REGION/SPLIT teacher labels.

    Training is explicit and opt-in. Production inference remains a small encoder.
    """

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
    ) -> str:
        if not examples:
            raise ValueError("At least one training example is required.")
        try:
            from torch.utils.data import DataLoader
            from sentence_transformers import CrossEncoder, InputExample
        except ImportError as exc:
            raise ImportError(
                "Fine-tuning requires sentence-transformers and torch."
            ) from exc

        train_rows = [
            InputExample(
                texts=[row.query, f"[LEFT]\n{row.left}\n[RIGHT]\n{row.right}"],
                label=float(row.label),
            )
            for row in examples
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
