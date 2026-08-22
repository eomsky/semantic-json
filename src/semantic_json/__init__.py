from .compiler import compile
from .repository import (
    SemanticRepository,
    SemanticMatch,
    EvidenceRegion,
    SearchResult,
    TRANSPORT_SCHEMA,
)
from .embeddings import LiteEmbedder, MultilingualE5Embedder
from .schemas import SemanticDocument, Proposition, Relation, Scope, SourceSpan

__version__ = "0.1.0a7"

__all__ = [
    "compile",
    "SemanticRepository",
    "SemanticMatch",
    "EvidenceRegion",
    "SearchResult",
    "TRANSPORT_SCHEMA",
    "LiteEmbedder",
    "MultilingualE5Embedder",
    "SemanticDocument",
    "Proposition",
    "Relation",
    "Scope",
    "SourceSpan",
]
