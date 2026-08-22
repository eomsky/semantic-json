from __future__ import annotations
import hashlib
import re
import unicodedata
import numpy as np

# 작은 한·영 의미 정규화 사전 (Small Korean/English semantic normalization lexicon)
# 이 사전은 의미 판단기가 아니라 경량 검색 recall을 돕는 장치입니다.
# (This lexicon assists lightweight retrieval recall; it is not a semantic judge.)
_CONCEPTS = {
    "repayment": ("상환", "원리금", "채무상환", "debt service", "debt servicing", "repayment", "principal and interest"),
    "capacity": ("상환능력", "채무상환능력", "repayment capacity", "debt service capacity"),
    "liquidity": ("유동성", "liquidity"),
    "cashflow": ("현금흐름", "현금창출력", "cash flow", "cash generation"),
    "revenue": ("매출", "매출액", "revenue", "sales"),
    "profit": ("수익성", "영업이익", "이익", "profitability", "operating profit", "earnings"),
    "debt": ("차입", "차입금", "부채", "borrowings", "debt", "leverage"),
    "maturity": ("만기", "maturity", "refinancing"),
    "uncertain": ("불확실", "단정하기 어렵", "단정하기는 어렵", "확인 필요", "uncertain", "difficult to conclude", "not confirmed"),
    "current": ("현재", "현재까지", "current", "currently"),
    "mediumterm": ("중장기", "중장기적", "medium term", "medium-term"),
    "future": ("내년", "향후", "예정", "전망", "next year", "future", "scheduled", "forecast"),
    "normal": ("정상", "양호", "satisfactory", "normal", "on schedule", "adequate"),
    "negative": ("악화", "저하", "감소", "deteriorate", "weaken", "decline"),
    "positive": ("개선", "증가", "양호", "improve", "increase", "positive"),
}


def _normalize(text: str) -> str:
    t = unicodedata.normalize("NFKC", text).lower()
    for concept, forms in _CONCEPTS.items():
        for form in sorted(forms, key=len, reverse=True):
            t = t.replace(form.lower(), f" concept_{concept} ")
    return re.sub(r"\s+", " ", t).strip()


def _features(text: str):
    t = _normalize(text)
    # 단어 + 문자 n-gram으로 한국어 조사/활용과 영문 표현 차이를 완화
    # (Word and character n-grams reduce sensitivity to Korean particles and wording.)
    words = re.findall(r"concept_[a-z]+|[a-z0-9]+|[가-힣]+", t)
    feats = ["w:" + w for w in words]
    compact = re.sub(r"\s+", "", t)
    feats += ["c:" + compact[i:i+3] for i in range(max(0, len(compact)-2))]
    return feats


class LiteEmbedder:
    """NumPy-only hashing embedder. No Torch, Transformers, model download, or GPU required."""

    def __init__(self, dimensions: int = 2048):
        self.dimensions = dimensions

    def _encode_one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dimensions, dtype=np.float32)
        for feat in _features(text):
            digest = hashlib.blake2b(feat.encode("utf-8"), digest_size=8).digest()
            h = int.from_bytes(digest, "little")
            idx = h % self.dimensions
            sign = 1.0 if ((h >> 63) & 1) == 0 else -1.0
            v[idx] += sign
        norm = float(np.linalg.norm(v))
        return v / norm if norm else v

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimensions), dtype=np.float32)
        return np.stack([self._encode_one(x) for x in texts])

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode_one(text)


class MultilingualE5Embedder:
    """Optional higher-quality multilingual embedding backend using sentence-transformers."""

    def __init__(self, model_name: str = "intfloat/multilingual-e5-small", device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError(
                'Install the optional Transformers backend with: '
                'pip install "semantic-json-transport[transformers]"'
            ) from e
        self.model_name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_passages(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            ["passage: " + x for x in texts],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )

    def encode_query(self, text: str) -> np.ndarray:
        return self.model.encode(
            ["query: " + text],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        )[0]
