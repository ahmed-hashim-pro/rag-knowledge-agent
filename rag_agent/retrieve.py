"""Vector store access, scored retrieval, and context assembly.

Two guardrails live in this module:

* **Confidence gating.** :func:`retrieve` drops every chunk whose cosine
  similarity falls below the configured floor. If nothing clears the floor the
  caller gets an empty list and refuses to answer — no model call is made.
* **Prompt-injection containment.** :func:`build_context` wraps each chunk in a
  ``<document>`` element and neutralises any closing tag inside the chunk text,
  so retrieved content cannot break out of its envelope and impersonate an
  instruction. The system prompt in :mod:`rag_agent.agent` is the other half of
  this defence.
"""

from __future__ import annotations

import contextlib
import os
import re
import sys
import tempfile
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from rag_agent.config import RagConfig, estimate_tokens
from rag_agent.ingest import Chunk

#: Any literal closing tag inside retrieved text is defanged before it reaches
#: the model, so a document cannot terminate its own envelope.
_CLOSING_TAG_RE = re.compile(r"</\s*document\s*>", re.IGNORECASE)

#: Lines the embedding backend emits on every single model load. Recognised so
#: they can be dropped without also hiding genuine errors.
_LOADER_NOISE_RE = re.compile(
    r"unauthenticated requests to the HF Hub|^Loading weights:|^Batches:"
)


@contextlib.contextmanager
def _quiet_model_load() -> Iterator[None]:
    """Suppress the embedding backend's loader chatter, but not real errors.

    ``sentence-transformers`` writes an HF Hub notice and a progress bar
    directly to file descriptor 2 from native code — not through ``warnings``
    or ``logging``, so neither can turn them off, and they would otherwise
    appear on every ``rag stats``. Capture fd 2 for the duration of the load and
    replay whatever is not recognised loader noise, so a genuine failure still
    reaches the user.

    Set ``RAG_VERBOSE_LOADER=1`` to disable this and see the raw output.
    """
    if os.environ.get("RAG_VERBOSE_LOADER"):
        yield
        return

    sys.stderr.flush()
    saved_fd = os.dup(2)
    captured = ""
    try:
        with tempfile.TemporaryFile() as buffer:
            os.dup2(buffer.fileno(), 2)
            try:
                yield
            finally:
                sys.stderr.flush()
                os.dup2(saved_fd, 2)
                buffer.seek(0)
                captured = buffer.read().decode("utf-8", errors="replace")
    finally:
        os.close(saved_fd)

    # splitlines() also breaks on \r, so each progress-bar frame is its own line.
    for line in captured.splitlines():
        if line.strip() and not _LOADER_NOISE_RE.search(line):
            print(line, file=sys.stderr)


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned from a similarity search, with its score."""

    text: str
    source: str
    heading: str
    chunk_index: int
    score: float

    @property
    def citation(self) -> str:
        """The ``[source:heading]`` label the model is asked to cite with."""
        return f"[{self.source}:{self.heading}]" if self.heading else f"[{self.source}]"


def _escape_document_text(text: str) -> str:
    return _CLOSING_TAG_RE.sub("<\\/document>", text)


def _attr(value: str) -> str:
    return value.replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")


class VectorStore:
    """Thin, typed wrapper over a persistent ChromaDB collection."""

    def __init__(
        self,
        config: RagConfig,
        embedding_function: Any | None = None,
    ) -> None:
        self.config = config
        self._embedding_function = embedding_function
        self._collection: Any | None = None

    # -- construction ------------------------------------------------------

    def _build_embedding_function(self) -> Any:
        # Set before the backend is imported: these turn off the download and
        # weight-loading progress bars at the source. The remaining HF notice is
        # printed from native code and is handled by _quiet_model_load below.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
        os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

        with _quiet_model_load():
            from chromadb.utils import embedding_functions

            return embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=self.config.embedding_model
            )

    @property
    def collection(self) -> Any:
        """The underlying Chroma collection, created on first use."""
        if self._collection is None:
            import chromadb
            from chromadb.config import Settings

            self.config.persist_dir.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(
                path=str(self.config.persist_dir),
                settings=Settings(anonymized_telemetry=False, allow_reset=True),
            )
            if self._embedding_function is None:
                self._embedding_function = self._build_embedding_function()
            self._collection = client.get_or_create_collection(
                name=self.config.collection_name,
                embedding_function=self._embedding_function,
                # Cosine space keeps distances in [0, 2] so that
                # ``similarity = 1 - distance`` is a well-behaved [-1, 1] score.
                metadata={"hnsw:space": "cosine"},
            )
        return self._collection

    @property
    def distance_space(self) -> str:
        """The distance metric the collection was actually created with.

        Worth asserting on: if a Chroma version ignored the space hint and fell
        back to L2, ``1 - distance`` would go negative and the score floor would
        silently invert.
        """
        metadata = getattr(self.collection, "metadata", None) or {}
        space = metadata.get("hnsw:space")
        if space:
            return str(space)
        configuration = getattr(self.collection, "configuration_json", None) or {}
        hnsw = configuration.get("hnsw") or configuration.get("spann") or {}
        return str(hnsw.get("space", "unknown"))

    # -- writes ------------------------------------------------------------

    def source_state(self, source: str) -> tuple[str | None, int]:
        """Return ``(indexed_file_hash, chunk_count)`` for ``source``."""
        existing = self.collection.get(where={"source": source}, include=["metadatas"])
        metadatas = existing.get("metadatas") or []
        if not metadatas:
            return None, 0
        return str(metadatas[0].get("file_hash", "")) or None, len(metadatas)

    def delete_source(self, source: str) -> None:
        self.collection.delete(where={"source": source})

    def replace_source(
        self, source: str, file_hash: str, chunks: Sequence[Chunk]
    ) -> None:
        """Delete any existing chunks for ``source`` and index ``chunks``."""
        self.delete_source(source)
        if not chunks:
            return
        self.collection.add(
            ids=[f"{source}::{chunk.index}" for chunk in chunks],
            documents=[chunk.text for chunk in chunks],
            metadatas=[
                {
                    "source": source,
                    # Chroma rejects ``None`` metadata values; a chunk that
                    # precedes any heading stores an empty string instead.
                    "heading": chunk.heading or "",
                    "chunk_index": chunk.index,
                    "file_hash": file_hash,
                    "chars": len(chunk.text),
                }
                for chunk in chunks
            ],
        )

    # -- reads -------------------------------------------------------------

    def count(self) -> int:
        return int(self.collection.count())

    def sources(self) -> dict[str, int]:
        """Map every indexed source path to its chunk count."""
        payload = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for metadata in payload.get("metadatas") or []:
            source = str(metadata.get("source", "unknown"))
            counts[source] = counts.get(source, 0) + 1
        return dict(sorted(counts.items()))

    def search(self, question: str, top_k: int) -> list[RetrievedChunk]:
        """Return the ``top_k`` nearest chunks, scored by cosine similarity."""
        if self.count() == 0:
            return []
        payload = self.collection.query(
            query_texts=[question],
            n_results=min(top_k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        # Chroma returns strictly parallel arrays; strict=True makes that
        # invariant fail loudly rather than silently truncating results.
        for text, metadata, distance in zip(
            documents, metadatas, distances, strict=True
        ):
            metadata = metadata or {}
            results.append(
                RetrievedChunk(
                    text=text,
                    source=str(metadata.get("source", "unknown")),
                    heading=str(metadata.get("heading", "")),
                    chunk_index=int(metadata.get("chunk_index", 0)),
                    # Chroma reports cosine *distance*; invert it so that a
                    # higher score always means a better match.
                    score=round(1.0 - float(distance), 4),
                )
            )
        return results


def retrieve(
    store: VectorStore,
    question: str,
    config: RagConfig,
    top_k: int | None = None,
    min_score: float | None = None,
) -> list[RetrievedChunk]:
    """Search, then apply the confidence floor.

    An empty result is the signal for the caller to refuse rather than guess.
    """
    k = top_k if top_k is not None else config.top_k
    floor = min_score if min_score is not None else config.min_score
    return [chunk for chunk in store.search(question, k) if chunk.score >= floor]


def build_context(
    chunks: Iterable[RetrievedChunk],
    max_context_tokens: int,
) -> tuple[str, list[RetrievedChunk]]:
    """Render chunks as ``<document>`` elements within a token budget.

    Returns the rendered block and the chunks that actually fit. A chunk is
    dropped whole rather than truncated, so every document the model sees is
    complete and every citation it can make points at something it read in full.
    """
    rendered: list[str] = []
    included: list[RetrievedChunk] = []
    used = 0

    for chunk in chunks:
        body = _escape_document_text(chunk.text)
        element = (
            f'<document index="{len(included) + 1}" '
            f'source="{_attr(chunk.source)}" '
            f'heading="{_attr(chunk.heading)}" '
            f'score="{chunk.score:.3f}">\n{body}\n</document>'
        )
        cost = estimate_tokens(element)
        if included and used + cost > max_context_tokens:
            break
        if not included and cost > max_context_tokens:
            # Never send zero context because the single best chunk is huge:
            # include it and let ``max_tokens`` on the request bound the rest.
            rendered.append(element)
            included.append(chunk)
            break
        rendered.append(element)
        included.append(chunk)
        used += cost

    return "\n\n".join(rendered), included
