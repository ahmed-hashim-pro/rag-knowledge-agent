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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from rag_agent.config import (
    RETRIEVAL_MODES,
    RagConfig,
    candidate_pool_size,
    estimate_tokens,
    normalise_bm25,
)
from rag_agent.ingest import Chunk
from rag_agent.lexical import (
    INDEX_FILENAME,
    BM25Index,
    corpus_fingerprint,
    reciprocal_rank_fusion,
)

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
    """One retrieved chunk and the evidence for keeping it.

    ``score`` is what the confidence floor is applied to and what is shown to
    the model: a [0, 1] support score, whatever mode produced the chunk. The
    per-retriever fields below are diagnostics — either may be ``None`` when
    only one retriever surfaced this chunk.
    """

    text: str
    source: str
    heading: str
    chunk_index: int
    score: float
    chunk_id: str = ""
    vector_score: float | None = None
    lexical_score: float | None = None
    fused_score: float | None = None

    @property
    def citation(self) -> str:
        """The ``[source:heading]`` label the model is asked to cite with."""
        return f"[{self.source}:{self.heading}]" if self.heading else f"[{self.source}]"

    @property
    def matched_by(self) -> str:
        """Which retrievers found this chunk — for ``--show-sources``."""
        found = []
        if self.vector_score is not None:
            found.append("vector")
        if self.lexical_score is not None:
            found.append("bm25")
        return "+".join(found) if found else "unknown"


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
        self._lexical_index: BM25Index | None = None

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
        ids = (payload.get("ids") or [[]])[0]
        documents = (payload.get("documents") or [[]])[0]
        metadatas = (payload.get("metadatas") or [[]])[0]
        distances = (payload.get("distances") or [[]])[0]

        results: list[RetrievedChunk] = []
        # Chroma returns strictly parallel arrays; strict=True makes that
        # invariant fail loudly rather than silently truncating results.
        for chunk_id, text, metadata, distance in zip(
            ids, documents, metadatas, distances, strict=True
        ):
            # Chroma reports cosine *distance*; invert it so higher is better.
            similarity = round(1.0 - float(distance), 4)
            results.append(
                _chunk_from(
                    chunk_id, text, metadata, score=similarity, vector=similarity
                )
            )
        return results

    def snapshot(self) -> tuple[list[str], list[str], list[str]]:
        """Every chunk as ``(ids, documents, file_hashes)``, for index building."""
        payload = self.collection.get(include=["documents", "metadatas"])
        ids = list(payload.get("ids") or [])
        documents = list(payload.get("documents") or [])
        hashes = [
            str((metadata or {}).get("file_hash", ""))
            for metadata in payload.get("metadatas") or []
        ]
        return ids, documents, hashes

    def fetch(self, chunk_ids: Sequence[str]) -> dict[str, RetrievedChunk]:
        """Hydrate chunks by id, for lexical hits the vector search never saw."""
        if not chunk_ids:
            return {}
        payload = self.collection.get(
            ids=list(chunk_ids), include=["documents", "metadatas"]
        )
        found: dict[str, RetrievedChunk] = {}
        for chunk_id, text, metadata in zip(
            payload.get("ids") or [],
            payload.get("documents") or [],
            payload.get("metadatas") or [],
            strict=True,
        ):
            found[chunk_id] = _chunk_from(chunk_id, text, metadata, score=0.0)
        return found

    # -- lexical index -----------------------------------------------------

    @property
    def lexical_index_path(self) -> Path:
        return self.config.persist_dir / INDEX_FILENAME

    def build_lexical_index(self) -> BM25Index:
        """Rebuild the BM25 index from the collection and persist it."""
        ids, documents, hashes = self.snapshot()
        index = BM25Index.build(
            ids=ids,
            documents=documents,
            fingerprint=corpus_fingerprint(ids, hashes),
        )
        index.save(self.lexical_index_path)
        return index

    def lexical_index(self, rebuild_if_stale: bool = True) -> BM25Index:
        """Load the BM25 index, rebuilding it if it is missing or out of date.

        The fingerprint check means an index left behind by an older ingest is
        silently corrected rather than quietly returning results for documents
        that no longer exist.
        """
        if self._lexical_index is not None:
            return self._lexical_index

        index = BM25Index.load(self.lexical_index_path)
        if index is not None and rebuild_if_stale:
            ids, _, hashes = self.snapshot()
            if index.fingerprint != corpus_fingerprint(ids, hashes):
                index = None
        if index is None:
            index = self.build_lexical_index()

        self._lexical_index = index
        return index


def _chunk_from(
    chunk_id: str,
    text: str,
    metadata: Any,
    score: float,
    vector: float | None = None,
    lexical: float | None = None,
    fused: float | None = None,
) -> RetrievedChunk:
    metadata = metadata or {}
    return RetrievedChunk(
        text=text,
        source=str(metadata.get("source", "unknown")),
        heading=str(metadata.get("heading", "")),
        chunk_index=int(metadata.get("chunk_index", 0)),
        score=score,
        chunk_id=chunk_id,
        vector_score=vector,
        lexical_score=lexical,
        fused_score=fused,
    )


def _vector_candidates(
    store: VectorStore, question: str, pool: int
) -> list[RetrievedChunk]:
    return store.search(question, pool)


def _lexical_candidates(
    store: VectorStore, question: str, pool: int, saturation: float
) -> list[RetrievedChunk]:
    hits = store.lexical_index().search(question, pool)
    if not hits:
        return []
    hydrated = store.fetch([chunk_id for chunk_id, _ in hits])
    results: list[RetrievedChunk] = []
    for chunk_id, raw in hits:
        base = hydrated.get(chunk_id)
        if base is None:
            continue
        results.append(
            replace(
                base,
                score=round(normalise_bm25(raw, saturation), 4),
                lexical_score=round(raw, 4),
            )
        )
    return results


def _fuse(
    vector_hits: Sequence[RetrievedChunk],
    lexical_hits: Sequence[RetrievedChunk],
    rrf_k: int,
    limit: int,
) -> list[RetrievedChunk]:
    """Merge two rankings with RRF, guaranteeing each retriever's best hit survives.

    RRF decides the *order*; the confidence floor is applied to ``score``, the
    best normalised support any single retriever gave the chunk. Fusion ranks
    are not comparable to a similarity threshold, so using them for gating
    would silently invalidate the calibrated floor.

    The anchoring matters more than it looks. RRF rewards agreement, so a chunk
    that only one retriever found is easily outscored by several that both
    found — including that retriever's own top hit. Measured on the sample
    corpus, unanchored fusion dropped the best semantic match for "why would a
    robot slow down after we loaded it" out of the top-5 entirely, taking the
    query's support score from 0.545 to 0.341 and turning an answerable question
    into a refusal. Anchoring makes hybrid a superset of what either mode alone
    would have surfaced.
    """
    fused_scores = reciprocal_rank_fusion(
        [
            [hit.chunk_id for hit in vector_hits],
            [hit.chunk_id for hit in lexical_hits],
        ],
        k=rrf_k,
    )

    merged: dict[str, RetrievedChunk] = {}
    for hit in (*vector_hits, *lexical_hits):
        existing = merged.get(hit.chunk_id)
        if existing is None:
            merged[hit.chunk_id] = hit
            continue
        merged[hit.chunk_id] = replace(
            existing,
            vector_score=(
                existing.vector_score
                if existing.vector_score is not None
                else hit.vector_score
            ),
            lexical_score=(
                existing.lexical_score
                if existing.lexical_score is not None
                else hit.lexical_score
            ),
            score=max(existing.score, hit.score),
        )

    combined = [
        replace(hit, fused_score=round(fused_scores.get(chunk_id, 0.0), 6))
        for chunk_id, hit in merged.items()
    ]
    combined.sort(key=lambda hit: (-(hit.fused_score or 0.0), -hit.score))

    anchors = [hits[0].chunk_id for hits in (vector_hits, lexical_hits) if hits]
    selected = combined[:limit]
    present = {hit.chunk_id for hit in selected}
    for anchor in anchors:
        if anchor in present:
            continue
        promoted = next(hit for hit in combined if hit.chunk_id == anchor)
        for position in range(len(selected) - 1, -1, -1):
            if selected[position].chunk_id not in anchors:
                selected.pop(position)
                break
        selected.append(promoted)
        present.add(anchor)

    selected.sort(key=lambda hit: (-(hit.fused_score or 0.0), -hit.score))
    return selected


def retrieve(
    store: VectorStore,
    question: str,
    config: RagConfig,
    top_k: int | None = None,
    min_score: float | None = None,
    mode: str | None = None,
) -> list[RetrievedChunk]:
    """Retrieve candidates in the requested mode, then apply the confidence floor.

    An empty result is the signal for the caller to refuse rather than guess.
    """
    k = top_k if top_k is not None else config.top_k
    floor = min_score if min_score is not None else config.min_score
    selected = mode if mode is not None else config.retrieval_mode
    if selected not in RETRIEVAL_MODES:
        raise ValueError(
            f"unknown retrieval mode {selected!r}; "
            f"expected one of {', '.join(RETRIEVAL_MODES)}"
        )

    if store.count() == 0:
        return []

    if selected == "vector":
        ranked = _vector_candidates(store, question, k)
    elif selected == "bm25":
        ranked = _lexical_candidates(store, question, k, config.bm25_saturation)
    else:
        pool = candidate_pool_size(config)
        ranked = _fuse(
            _vector_candidates(store, question, pool),
            _lexical_candidates(store, question, pool, config.bm25_saturation),
            rrf_k=config.rrf_k,
            limit=k,
        )

    return [chunk for chunk in ranked[:k] if chunk.score >= floor]


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
