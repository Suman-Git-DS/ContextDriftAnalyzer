"""
context-drift-analyzer: Detect, explain, and solve context drift in LLM conversations.

Provides:
- Drift scoring (0-100) with semantic embeddings
- Drift explanation (why did it drift?)
- Context management (original context + session summaries)
- Persistence across restarts (.session_memory files)
- On-demand or always-on modes
- CLI tool for monitoring
"""

from context_drift_analyzer.core.session import Session, FewShotExample
from context_drift_analyzer.core.analyzer import DriftAnalyzer
from context_drift_analyzer.core.scorer import DriftScore, DriftVerdict
from context_drift_analyzer.tracker import DriftTracker, TurnResult, DriftReport
from context_drift_analyzer.context.manager import ContextManager, SessionSummary
from context_drift_analyzer.context.explainer import DriftExplainer
from context_drift_analyzer.persistence.session_memory import SessionMemoryStore
from context_drift_analyzer.strategies.keyword import KeywordStrategy
from context_drift_analyzer.strategies.token_overlap import TokenOverlapStrategy
from context_drift_analyzer.strategies.composite import CompositeStrategy
from context_drift_analyzer.strategies.embedding_base import EmbeddingStrategy
from context_drift_analyzer.strategies.callable_embedding import CallableEmbeddingStrategy
from context_drift_analyzer.wrap import wrap, DriftClientWrapper

__version__ = "0.5.0"

__all__ = [
    # Main entry point
    "DriftTracker",
    "TurnResult",
    "DriftReport",
    # Core
    "Session",
    "FewShotExample",
    "DriftAnalyzer",
    "DriftScore",
    "DriftVerdict",
    # Context management
    "ContextManager",
    "SessionSummary",
    "DriftExplainer",
    # Persistence
    "SessionMemoryStore",
    # Client wrapper
    "wrap",
    "DriftClientWrapper",
    # Strategies
    "KeywordStrategy",
    "TokenOverlapStrategy",
    "CompositeStrategy",
    "EmbeddingStrategy",
    "CallableEmbeddingStrategy",
    "__version__",
]


# Lazy imports for optional dependencies
def __getattr__(name: str):
    if name == "SentenceTransformerStrategy":
        from context_drift_analyzer.strategies.sentence_transformer import (
            SentenceTransformerStrategy,
        )
        return SentenceTransformerStrategy
    if name == "OpenAIEmbeddingStrategy":
        from context_drift_analyzer.strategies.openai_embedding import (
            OpenAIEmbeddingStrategy,
        )
        return OpenAIEmbeddingStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
