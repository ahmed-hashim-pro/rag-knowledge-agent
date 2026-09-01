"""Document loading, markdown-aware chunking, and idempotent indexing.

Ingestion is content-addressed: every file is hashed, and the hash is stored on
each of its chunks. Re-running ``rag ingest`` over an unchanged corpus is a no-op
that touches no embeddings; a file whose bytes changed has its old chunks deleted
and replaced atomically-enough for a local index.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from rag_agent.config import SUPPORTED_SUFFIXES, RagConfig, estimate_tokens

if TYPE_CHECKING:  # pragma: no cover - typing only
    from rag_agent.retrieve import VectorStore

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_SENTENCE_END_RE = re.compile(r"(?<=[.!?])\s+")

#: Separator between levels of a heading path, e.g. ``"Meridian-3 AMR > Power"``.
HEADING_SEPARATOR = " > "


@dataclass(frozen=True)
class Chunk:
    """A slice of one document, with the heading path it was found under."""

    text: str
    heading: str
    index: int


@dataclass(frozen=True)
class FileResult:
    """Outcome of ingesting a single file."""

    source: str
    status: str  # indexed | updated | skipped | empty | error
    chunks: int
    detail: str = ""


@dataclass(frozen=True)
class IngestSummary:
    """Aggregate outcome of an ingest run."""

    results: tuple[FileResult, ...]

    @property
    def total_chunks(self) -> int:
        return sum(r.chunks for r in self.results)

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def iter_source_files(target: Path) -> Iterator[Path]:
    """Yield every supported file under ``target``, sorted, skipping dot-dirs."""
    if target.is_file():
        if target.suffix.lower() in SUPPORTED_SUFFIXES:
            yield target
        return
    for path in sorted(target.rglob("*")):
        if not path.is_file():
            continue
        if any(part.startswith(".") for part in path.relative_to(target).parts):
            continue
        if path.suffix.lower() in SUPPORTED_SUFFIXES:
            yield path


def _load_pdf(path: Path) -> str:
    """Extract text from a PDF, one markdown ``## Page N`` section per page.

    Emitting page headings means PDF chunks carry a meaningful heading in their
    citation (``[handbook.pdf:Page 4]``) even though PDFs have no markdown
    structure of their own.
    """
    from pypdf import PdfReader  # imported lazily: only needed for PDFs

    reader = PdfReader(str(path))
    parts: list[str] = []
    for number, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if text:
            parts.append(f"## Page {number}\n\n{text}")
    return "\n\n".join(parts)


def load_document(path: Path) -> str:
    """Read ``path`` into markdown-ish text, dispatching on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return _load_pdf(path)
    if suffix in (".md", ".markdown", ".txt"):
        return path.read_text(encoding="utf-8", errors="replace")
    raise ValueError(f"Unsupported file type: {path.suffix}")


def file_digest(path: Path) -> str:
    """Return the SHA-256 of a file's bytes, used for change detection."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def relative_source(path: Path, root: Path | None = None) -> str:
    """Render a repo-relative citation label for ``path``.

    Never returns an absolute path or a ``..`` escape: a file outside both the
    working directory and the ingest root degrades to its bare filename. This
    keeps local filesystem layout out of citations, indexes, and CLI output.
    """
    resolved = path.resolve()
    bases: list[Path] = [Path.cwd()]
    if root is not None:
        root_resolved = root.resolve()
        base_dir = root_resolved if root_resolved.is_dir() else root_resolved.parent
        # The parent first, so an ingest of ``sample_corpus/`` run from
        # elsewhere still labels chunks ``sample_corpus/faq.md`` rather than
        # the bare ``faq.md``.
        bases.append(base_dir.parent)
        bases.append(base_dir)
    for base in bases:
        try:
            return resolved.relative_to(base).as_posix()
        except ValueError:
            continue
    return resolved.name


# --------------------------------------------------------------------------
# Chunking
# --------------------------------------------------------------------------


def _iter_sections(text: str) -> Iterator[tuple[str, str]]:
    """Split markdown into ``(heading_path, body)`` pairs.

    Headings inside fenced code blocks are ignored, so a ``# comment`` line in a
    shell snippet does not silently start a new section.
    """
    stack: list[tuple[int, str]] = []
    body: list[str] = []
    in_fence = False
    fence_marker = ""

    def heading_path() -> str:
        return HEADING_SEPARATOR.join(title for _, title in stack)

    for line in text.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            body.append(line)
            continue

        match = None if in_fence else _HEADING_RE.match(line)
        if match is None:
            body.append(line)
            continue

        chunk_body = "\n".join(body).strip()
        if chunk_body:
            yield heading_path(), chunk_body
        body = []

        level = len(match.group(1))
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, match.group(2).strip()))

    tail = "\n".join(body).strip()
    if tail:
        yield heading_path(), tail


def _iter_paragraphs(body: str) -> Iterator[str]:
    """Split a section body on blank lines, keeping fenced code blocks whole."""
    current: list[str] = []
    in_fence = False
    fence_marker = ""

    for line in body.splitlines():
        fence = _FENCE_RE.match(line)
        if fence:
            marker = fence.group(1)
            if not in_fence:
                in_fence, fence_marker = True, marker
            elif marker == fence_marker:
                in_fence, fence_marker = False, ""
            current.append(line)
            continue
        if not line.strip() and not in_fence:
            block = "\n".join(current).strip()
            if block:
                yield block
            current = []
            continue
        current.append(line)

    block = "\n".join(current).strip()
    if block:
        yield block


def _split_oversized(paragraph: str, max_tokens: int) -> list[str]:
    """Break a single over-long paragraph on sentence, then hard, boundaries."""
    if estimate_tokens(paragraph) <= max_tokens:
        return [paragraph]

    pieces: list[str] = []
    buffer = ""
    for sentence in _SENTENCE_END_RE.split(paragraph):
        candidate = f"{buffer} {sentence}".strip() if buffer else sentence
        if buffer and estimate_tokens(candidate) > max_tokens:
            pieces.append(buffer)
            buffer = sentence
        else:
            buffer = candidate
    if buffer:
        pieces.append(buffer)

    # A single sentence can still exceed the budget (tables, minified blobs).
    limit = max_tokens * 4
    hard_split: list[str] = []
    for piece in pieces:
        while len(piece) > limit:
            hard_split.append(piece[:limit])
            piece = piece[limit:]
        if piece:
            hard_split.append(piece)
    return hard_split


def _overlap_tail(paragraphs: Sequence[str], overlap_tokens: int) -> list[str]:
    """Pick the trailing paragraphs of a chunk to repeat in the next chunk."""
    if overlap_tokens <= 0 or not paragraphs:
        return []
    tail: list[str] = []
    budget = overlap_tokens
    for paragraph in reversed(paragraphs):
        cost = estimate_tokens(paragraph)
        if cost > budget:
            if not tail:
                # Repeat the tail characters of an over-long final paragraph,
                # snapped to a word boundary so the overlap reads cleanly.
                slice_chars = overlap_tokens * 4
                snippet = paragraph[-slice_chars:]
                _, _, remainder = snippet.partition(" ")
                tail.append(remainder or snippet)
            break
        tail.append(paragraph)
        budget -= cost
    tail.reverse()
    return tail


def chunk_markdown(
    text: str,
    chunk_tokens: int,
    overlap_tokens: int,
) -> list[Chunk]:
    """Chunk markdown into ~``chunk_tokens`` pieces that respect its structure.

    Sections never bleed into each other: a chunk belongs to exactly one heading
    path, which is what makes ``[source:heading]`` citations precise. Within a
    section, paragraphs are packed greedily and consecutive chunks overlap by
    roughly ``overlap_tokens`` so a fact split across a boundary is still
    retrievable from either side.
    """
    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    chunks: list[Chunk] = []
    for heading, body in _iter_sections(text):
        chunks.extend(
            _pack_section(
                heading=heading,
                body=body,
                chunk_tokens=chunk_tokens,
                overlap_tokens=overlap_tokens,
                start_index=len(chunks),
            )
        )
    return chunks


def _pack_section(
    heading: str,
    body: str,
    chunk_tokens: int,
    overlap_tokens: int,
    start_index: int,
) -> list[Chunk]:
    """Greedily pack one section's paragraphs into overlapping chunks."""
    chunks: list[Chunk] = []
    index = start_index
    buffer: list[str] = []
    buffer_tokens = 0

    def flush(carry: bool) -> None:
        nonlocal buffer, buffer_tokens, index
        if not buffer:
            return
        joined = "\n\n".join(buffer).strip()
        if joined:
            chunks.append(Chunk(text=joined, heading=heading, index=index))
            index += 1
        carried = _overlap_tail(buffer, overlap_tokens) if carry else []
        buffer = list(carried)
        buffer_tokens = sum(estimate_tokens(part) for part in buffer)

    for paragraph in _iter_paragraphs(body):
        for piece in _split_oversized(paragraph, chunk_tokens):
            cost = estimate_tokens(piece)
            if buffer and buffer_tokens + cost > chunk_tokens:
                flush(carry=True)
            buffer.append(piece)
            buffer_tokens += cost

    flush(carry=False)
    return chunks


def chunk_document(path: Path, config: RagConfig) -> list[Chunk]:
    """Load and chunk a single document."""
    return chunk_markdown(
        load_document(path),
        chunk_tokens=config.chunk_tokens,
        overlap_tokens=config.chunk_overlap_tokens,
    )


# --------------------------------------------------------------------------
# Indexing
# --------------------------------------------------------------------------


def ingest_path(
    target: Path,
    store: VectorStore,
    config: RagConfig,
    on_file: Callable[[FileResult], None] | None = None,
    force: bool = False,
) -> IngestSummary:
    """Index every supported file under ``target`` into ``store``.

    Idempotent: a file whose SHA-256 already matches the indexed copy is skipped
    without re-embedding. Pass ``force=True`` to re-embed regardless.
    """
    if not target.exists():
        raise FileNotFoundError(f"No such file or directory: {target}")

    results: list[FileResult] = []
    for path in iter_source_files(target):
        result = _ingest_file(path, target, store, config, force=force)
        results.append(result)
        if on_file is not None:
            on_file(result)

    # Rebuilt unconditionally: the BM25 index is derived state, and a stale one
    # would return hits for chunks the collection no longer holds.
    if results:
        store.build_lexical_index()

    return IngestSummary(results=tuple(results))


def _ingest_file(
    path: Path,
    root: Path,
    store: VectorStore,
    config: RagConfig,
    force: bool,
) -> FileResult:
    source = relative_source(path, root)
    try:
        digest = file_digest(path)
    except OSError as exc:
        return FileResult(source, "error", 0, str(exc))

    existing_hash, existing_count = store.source_state(source)
    if not force and existing_hash == digest:
        return FileResult(source, "skipped", existing_count, "unchanged")

    try:
        chunks = chunk_document(path, config)
    except Exception as exc:  # noqa: BLE001 - one bad file must not abort a run
        return FileResult(source, "error", 0, f"{type(exc).__name__}: {exc}")

    if not chunks:
        if existing_hash is not None:
            store.delete_source(source)
        return FileResult(source, "empty", 0, "no extractable text")

    store.replace_source(source=source, file_hash=digest, chunks=chunks)
    status = "updated" if existing_hash is not None else "indexed"
    return FileResult(source, status, len(chunks))


def total_tokens(chunks: Iterable[Chunk]) -> int:
    """Estimated token total across ``chunks`` (used by ``rag stats``)."""
    return sum(estimate_tokens(chunk.text) for chunk in chunks)
