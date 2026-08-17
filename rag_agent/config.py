"""Configuration for the RAG knowledge agent.

Every tunable lives here so that the retrieval, generation, and guardrail
behaviour of the agent can be reasoned about from a single place. Values are
resolved from (in order of precedence) explicit CLI flags, environment
variables, then the defaults below.

The Anthropic API key is deliberately *not* part of :class:`RagConfig`. It is
read from the ``ANTHROPIC_API_KEY`` environment variable at call time and is
never persisted, logged, or written to disk.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

#: Anthropic model used for answer generation.
DEFAULT_MODEL = "claude-sonnet-4-6"

#: Sentence-transformers model used for embeddings. Runs fully locally, so
#: indexing a corpus costs nothing and requires no API key.
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

DEFAULT_COLLECTION = "knowledge"
DEFAULT_PERSIST_DIR = "./chroma_db"

#: Target chunk size and overlap, in estimated tokens.
DEFAULT_CHUNK_TOKENS = 500
DEFAULT_CHUNK_OVERLAP_TOKENS = 50

#: Retrieval defaults.
DEFAULT_TOP_K = 5

#: Cosine-similarity floor a chunk must clear to count as supporting evidence.
#:
#: Measured, not guessed. Against the bundled sample corpus (19 probe questions,
#: see docs/design-notes.md) the best chunk for an on-topic question scored
#: 0.457-0.760, and for an off-topic question 0.115-0.436. A floor of 0.35
#: refuses 6 of 7 off-topic questions while clearing every on-topic one by at
#: least 0.107 — deliberately favouring the safer error, since a false refusal
#: on an answerable question is worse than passing a weak chunk to a model that
#: is separately instructed to decline unsupported questions.
DEFAULT_MIN_SCORE = 0.35

#: Hard ceiling on the retrieved context handed to the model, in estimated
#: tokens. Chunks past the budget are dropped rather than truncated, so a
#: citation never points at a document the model only partially saw.
DEFAULT_MAX_CONTEXT_TOKENS = 6000

#: Ceiling on generated output.
DEFAULT_MAX_OUTPUT_TOKENS = 1500

#: Number of prior conversation turns (user + assistant pairs) replayed as
#: message history in ``rag chat``.
DEFAULT_HISTORY_TURNS = 10

#: Ceiling on replayed history, in estimated tokens. Each historical turn keeps
#: the retrieved context it was answered from, so that citations in earlier
#: answers still refer to documents the model can see. That is coherent but not
#: free, so the oldest turns are dropped once this budget is exceeded.
DEFAULT_MAX_HISTORY_TOKENS = 12000

#: File extensions the ingester will read.
SUPPORTED_SUFFIXES = (".md", ".markdown", ".txt", ".pdf")

#: Rough characters-per-token ratio for English prose. Used for chunk sizing
#: and the context budget. This is an estimate, not a tokenizer: it never
#: needs the network and it errs on the side of over-counting, which keeps the
#: budget conservative.
CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate the token count of ``text`` without calling out to a tokenizer.

    Deliberately approximate. Used only for chunk sizing and for capping the
    context we send to the API, both of which want a cheap, offline, slightly
    pessimistic number rather than an exact one.
    """
    if not text:
        return 0
    return max(1, -(-len(text) // CHARS_PER_TOKEN))


def _env_str(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be an integer, got {raw!r}") from exc


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"{name} must be a number, got {raw!r}") from exc


@dataclass(frozen=True)
class RagConfig:
    """Resolved settings for a single CLI invocation."""

    persist_dir: Path = Path(DEFAULT_PERSIST_DIR)
    collection_name: str = DEFAULT_COLLECTION
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    model: str = DEFAULT_MODEL
    chunk_tokens: int = DEFAULT_CHUNK_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP_TOKENS
    top_k: int = DEFAULT_TOP_K
    min_score: float = DEFAULT_MIN_SCORE
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    history_turns: int = DEFAULT_HISTORY_TURNS
    max_history_tokens: int = DEFAULT_MAX_HISTORY_TOKENS

    def __post_init__(self) -> None:
        if self.chunk_overlap_tokens >= self.chunk_tokens:
            raise ValueError(
                "chunk_overlap_tokens must be smaller than chunk_tokens "
                f"(got {self.chunk_overlap_tokens} >= {self.chunk_tokens})"
            )
        if self.top_k < 1:
            raise ValueError(f"top_k must be at least 1, got {self.top_k}")
        if not 0.0 <= self.min_score <= 1.0:
            raise ValueError(
                f"min_score must be between 0.0 and 1.0, got {self.min_score}"
            )

    @classmethod
    def from_env(cls) -> RagConfig:
        """Build a config from ``RAG_*`` environment variables and defaults."""
        return cls(
            persist_dir=Path(_env_str("RAG_PERSIST_DIR", DEFAULT_PERSIST_DIR)),
            collection_name=_env_str("RAG_COLLECTION", DEFAULT_COLLECTION),
            embedding_model=_env_str("RAG_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL),
            model=_env_str("RAG_MODEL", DEFAULT_MODEL),
            chunk_tokens=_env_int("RAG_CHUNK_TOKENS", DEFAULT_CHUNK_TOKENS),
            chunk_overlap_tokens=_env_int(
                "RAG_CHUNK_OVERLAP_TOKENS", DEFAULT_CHUNK_OVERLAP_TOKENS
            ),
            top_k=_env_int("RAG_TOP_K", DEFAULT_TOP_K),
            min_score=_env_float("RAG_MIN_SCORE", DEFAULT_MIN_SCORE),
            max_context_tokens=_env_int(
                "RAG_MAX_CONTEXT_TOKENS", DEFAULT_MAX_CONTEXT_TOKENS
            ),
            max_output_tokens=_env_int(
                "RAG_MAX_OUTPUT_TOKENS", DEFAULT_MAX_OUTPUT_TOKENS
            ),
            history_turns=_env_int("RAG_HISTORY_TURNS", DEFAULT_HISTORY_TURNS),
            max_history_tokens=_env_int(
                "RAG_MAX_HISTORY_TOKENS", DEFAULT_MAX_HISTORY_TOKENS
            ),
        )

    def with_overrides(self, **overrides: object) -> RagConfig:
        """Return a copy with any non-``None`` overrides applied."""
        applied = {k: v for k, v in overrides.items() if v is not None}
        return replace(self, **applied)  # type: ignore[arg-type]


class MissingAPIKeyError(RuntimeError):
    """Raised when generation is attempted without ``ANTHROPIC_API_KEY`` set."""


def require_api_key() -> str:
    """Return the Anthropic API key from the environment, or explain why not.

    The key is read on demand and never stored. Indexing and retrieval do not
    call this, so the whole ``rag ingest`` / ``rag stats`` path works without
    any credentials at all.
    """
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not key:
        raise MissingAPIKeyError(
            "ANTHROPIC_API_KEY is not set. Export it before asking questions:\n"
            "    export ANTHROPIC_API_KEY=sk-ant-...\n"
            "Indexing (`rag ingest`) and `rag stats` do not need a key."
        )
    return key
