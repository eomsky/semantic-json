from __future__ import annotations

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class SemanticUnit:
    """A small, contiguous source span used as a retrieval/composition primitive."""

    id: str
    document_id: str
    start_char: int
    end_char: int
    start_line: int
    end_line: int
    text: str
    paragraph_id: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "document_id": self.document_id,
            "source": {
                "start_char": self.start_char,
                "end_char": self.end_char,
                "start_line": self.start_line,
                "end_line": self.end_line,
                "paragraph_id": self.paragraph_id,
            },
            "text": self.text,
        }


class FineGrainedUnitizer:
    """Deterministic, source-grounded micro-unitizer.

    The unitizer intentionally avoids semantic rewriting. It creates spans smaller than
    ordinary sentences when punctuation or strong clause markers provide a safe split.
    Unit boundaries are primitives for retrieval, not final EvidenceRegion boundaries.
    """

    _sentence_boundary = re.compile(r"(?<=[.!?。！？])\s+|\n+")
    _clause_boundary = re.compile(
        r"(?P<sep>[,;])\s+|"
        r"(?<=며)\s+|(?<=으나)\s+|(?<=지만)\s+|(?<=는데)\s+|"
        r"(?<=however,)\s+|(?<=therefore,)\s+",
        re.IGNORECASE,
    )

    @staticmethod
    def _line_number(text: str, offset: int) -> int:
        return text.count("\n", 0, max(0, offset)) + 1

    def _split_span(self, text: str, start: int, end: int) -> list[tuple[int, int]]:
        raw = text[start:end]
        pieces: list[tuple[int, int]] = []
        local_start = 0
        for match in self._clause_boundary.finditer(raw):
            cut = match.end() if match.groupdict().get("sep") else match.start()
            if cut > local_start:
                pieces.append((start + local_start, start + cut))
            local_start = match.end()
        if local_start < len(raw):
            pieces.append((start + local_start, end))
        return pieces or [(start, end)]

    def unitize(self, text: str, *, document_id: str) -> list[SemanticUnit]:
        sentence_spans: list[tuple[int, int]] = []
        cursor = 0
        for match in self._sentence_boundary.finditer(text):
            if match.start() > cursor:
                sentence_spans.append((cursor, match.start()))
            cursor = match.end()
        if cursor < len(text):
            sentence_spans.append((cursor, len(text)))

        units: list[SemanticUnit] = []
        paragraph_id = 0
        for sent_start, sent_end in sentence_spans:
            raw_sentence = text[sent_start:sent_end]
            if not raw_sentence.strip():
                continue
            if "\n\n" in text[max(0, sent_start - 2):sent_start + 1]:
                paragraph_id += 1
            for piece_start, piece_end in self._split_span(text, sent_start, sent_end):
                piece = text[piece_start:piece_end]
                left_trim = len(piece) - len(piece.lstrip())
                right_trim = len(piece) - len(piece.rstrip())
                start = piece_start + left_trim
                end = piece_end - right_trim
                if start >= end:
                    continue
                uid = f"U{len(units) + 1}"
                units.append(
                    SemanticUnit(
                        id=uid,
                        document_id=document_id,
                        start_char=start,
                        end_char=end,
                        start_line=self._line_number(text, start),
                        end_line=self._line_number(text, max(start, end - 1)),
                        text=text[start:end],
                        paragraph_id=paragraph_id,
                    )
                )
        return units
