"""
context-decay-drift: Detect, explain, and solve context drift in LLM conversations.

Provides:
- Drift scoring (0-100) with semantic embeddings
- Drift explanation (why did it drift?)
- Context management (original context + session summaries)
- Persistence across restarts (.session_memory files)
- On-demand or always-on modes
- CLI tool for monitoring
"""

from context_decay_drift.core.session import Session, FewShotExample
from context_decay_drift.core.analyzer import DriftAnalyzer
from context_decay_drift.core.scorer import DriftScore, DriftVerdict
from context_decay_drift.tracker import DriftTracker, TurnResult, DriftReport
from context_decay_drift.context.manager import ContextManager, SessionSummary
from context_decay_drift.context.explainer import DriftExplainer
from context_decay_drift.persistence.session_memory import SessionMemoryStore
from context_decay_drift.strategies.keyword import KeywordStrategy
from context_decay_drift.strategies.token_overlap import TokenOverlapStrategy
from context_decay_drift.strategies.composite import CompositeStrategy
from context_decay_drift.strategies.embedding_base import EmbeddingStrategy
from context_decay_drift.strategies.callable_embedding import CallableEmbeddingStrategy

__version__ = "0.3.0"

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
        from context_decay_drift.strategies.sentence_transformer import (
            SentenceTransformerStrategy,
        )
        return SentenceTransformerStrategy
    if name == "OpenAIEmbeddingStrategy":
        from context_decay_drift.strategies.openai_embedding import (
            OpenAIEmbeddingStrategy,
        )
        return OpenAIEmbeddingStrategy
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
