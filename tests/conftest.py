"""Shared fixtures.

Two embedding strategies are used deliberately:

* :class:`HashingEmbeddingFunction` — deterministic, instant, no model download.
  Used wherever the test is about *bookkeeping* (idempotency, metadata, chunk
  replacement) rather than semantics.
* the real ``all-MiniLM-L6-v2`` model — used only by the retrieval-relevance
  tests, because a fake embedder cannot demonstrate that relevant text ranks
  above irrelevant text. It runs locally and needs no API key; the first run
  downloads roughly 90 MB.

No test requires ``ANTHROPIC_API_KEY``: the Anthropic client is always stubbed.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any

import pytest
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings

from rag_agent.config import RagConfig

_EMBED_DIM = 64


class HashingEmbeddingFunction(EmbeddingFunction[Documents]):
    """A deterministic bag-of-words hashing embedder.

    Not semantic — identical text embeds identically and different text embeds
    differently, which is all the bookkeeping tests need.
    """

    def __init__(self, dim: int = _EMBED_DIM) -> None:
        self.dim = dim

    @staticmethod
    def name() -> str:
        return "test-hashing"

    def get_config(self) -> dict[str, Any]:
        return {"dim": _EMBED_DIM}

    @staticmethod
    def build_from_config(config: dict[str, Any]) -> HashingEmbeddingFunction:
        return HashingEmbeddingFunction(dim=config.get("dim", _EMBED_DIM))

    def __call__(self, input: Documents) -> Embeddings:  # noqa: A002
        vectors: list[list[float]] = []
        for document in input:
            vector = [0.0] * self.dim
            for token in document.lower().split():
                digest = hashlib.md5(token.encode("utf-8")).digest()
                vector[digest[0] % self.dim] += 1.0
            norm = math.sqrt(sum(value * value for value in vector)) or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


@pytest.fixture
def config(tmp_path: Path) -> RagConfig:
    """A config pointed at a throwaway index directory."""
    return RagConfig(
        persist_dir=tmp_path / "chroma_db",
        collection_name="test",
        chunk_tokens=120,
        chunk_overlap_tokens=20,
        top_k=5,
        min_score=0.0,
    )


@pytest.fixture
def fake_embedder() -> HashingEmbeddingFunction:
    return HashingEmbeddingFunction()


@pytest.fixture
def store(config: RagConfig, fake_embedder: HashingEmbeddingFunction):
    from rag_agent.retrieve import VectorStore

    return VectorStore(config, embedding_function=fake_embedder)


@pytest.fixture(scope="session")
def real_embedder():
    """The production sentence-transformers embedder, shared across tests."""
    from chromadb.utils import embedding_functions

    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-MiniLM-L6-v2"
    )


@pytest.fixture
def corpus(tmp_path: Path) -> Path:
    """A tiny three-document corpus with clearly separated topics."""
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "battery.md").write_text(
        "# Battery\n\n"
        "## Charging\n\n"
        "The robot charges from 20 percent to 80 percent in 45 minutes on a "
        "docking station. It returns to charge automatically below 25 percent "
        "state of charge.\n",
        encoding="utf-8",
    )
    (root / "network.md").write_text(
        "# Networking\n\n"
        "## Offline behaviour\n\n"
        "When the robot loses its network link it keeps working from its "
        "on-board task buffer for twelve minutes, then parks safely and waits "
        "for the connection to return.\n",
        encoding="utf-8",
    )
    (root / "baking.md").write_text(
        "# Sourdough\n\n"
        "## Starter\n\n"
        "Feed the starter equal weights of flour and water every morning and "
        "let it rise at room temperature until doubled before baking bread.\n",
        encoding="utf-8",
    )
    return root


# --------------------------------------------------------------------------
# Anthropic client stubs
# --------------------------------------------------------------------------


class StubTextBlock:
    type = "text"

    def __init__(self, text: str) -> None:
        self.text = text


class StubResponse:
    def __init__(self, text: str, stop_reason: str = "end_turn") -> None:
        self.content = [StubTextBlock(text)]
        self.stop_reason = stop_reason


class StubMessages:
    def __init__(self, replies: list[str]) -> None:
        self._replies = list(replies)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> StubResponse:
        self.calls.append(kwargs)
        if not self._replies:
            raise AssertionError("stub client received more calls than replies")
        return StubResponse(self._replies.pop(0))


class StubAnthropic:
    """Stands in for ``anthropic.Anthropic``; returns canned replies in order."""

    def __init__(self, *replies: str) -> None:
        self.messages = StubMessages(list(replies))

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self.messages.calls
