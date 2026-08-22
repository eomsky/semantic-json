from .compiler import compile
from .repository import (
    SemanticRepository,
    SemanticMatch,
    UnitMatch,
    BoundaryDecision,
    EvidenceRegion,
    SearchResult,
    SourceDocument,
    TRANSPORT_SCHEMA,
)
from .embeddings import LiteEmbedder, MultilingualE5Embedder
from .units import SemanticUnit, FineGrainedUnitizer
from .region_models import (
    RegionCompatibilityModel,
    RegionCompatibilityEncoder,
    HeuristicRegionModel,
    DEFAULT_REGION_MODEL,
)
from .training import (
    RegionTrainingExample,
    LLMRegionTeacher,
    RegionDatasetBuilder,
    RegionEncoderTrainer,
)
from .schemas import SemanticDocument, Proposition, Relation, Scope, SourceSpan

__version__ = "0.2.0a1"

__all__ = [
    "compile",
    "SemanticRepository",
    "SemanticMatch",
    "UnitMatch",
    "BoundaryDecision",
    "EvidenceRegion",
    "SearchResult",
    "SourceDocument",
    "TRANSPORT_SCHEMA",
    "SemanticUnit",
    "FineGrainedUnitizer",
    "RegionCompatibilityModel",
    "RegionCompatibilityEncoder",
    "HeuristicRegionModel",
    "DEFAULT_REGION_MODEL",
    "RegionTrainingExample",
    "LLMRegionTeacher",
    "RegionDatasetBuilder",
    "RegionEncoderTrainer",
    "LiteEmbedder",
    "MultilingualE5Embedder",
    "SemanticDocument",
    "Proposition",
    "Relation",
    "Scope",
    "SourceSpan",
]
