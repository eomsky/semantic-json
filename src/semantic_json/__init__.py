from .compiler import compile
from .repository import SemanticRepository, SemanticMatch
from .embeddings import LiteEmbedder, MultilingualE5Embedder
from .schemas import SemanticDocument, Proposition, Relation, Scope, SourceSpan

__version__ = "0.1.0a4"

__all__ = [
    "compile", "SemanticRepository", "SemanticMatch",
    "LiteEmbedder", "MultilingualE5Embedder",
    "SemanticDocument", "Proposition", "Relation", "Scope", "SourceSpan"
]
