from __future__ import annotations
import numpy as np

class MultilingualE5Embedder:
    """CPU 사용을 기본으로 하는 다국어 임베더 (Multilingual embedder, CPU by default)."""
    def __init__(self, model_name: str="intfloat/multilingual-e5-small", device: str="cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:
            raise ImportError('Install semantic search support with: pip install "semantic-json[search]"') from e
        self.model_name=model_name
        self.model=SentenceTransformer(model_name,device=device)

    def encode_passages(self,texts: list[str]) -> np.ndarray:
        return self.model.encode(["passage: "+x for x in texts],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)

    def encode_query(self,text: str) -> np.ndarray:
        return self.model.encode(["query: "+text],normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False)[0]
