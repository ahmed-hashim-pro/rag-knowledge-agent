"""Command line interface: ``rag ingest | ask | chat | stats``.

Output discipline: in ``--json`` mode **stdout carries the JSON document and
nothing else**. Progress, warnings, and retry notices always go to stderr, so
``rag ask --json "..." | jq .`` works unconditionally.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from rag_agent import __version__
from rag_agent.agent import (
    AnswerResult,
    InvalidJSONResponseError,
    RagAgent,
    Turn,
)
from rag_agent.config import MissingAPIKeyError, RagConfig
from rag_agent.ingest import FileResult, ingest_path
from rag_agent.retrieve import VectorStore

CHAT_BANNER = """\
rag chat — ask questions against the local index.
Commands: /exit to quit, /reset to clear conversation memory, /stats for index info.
"""


def _err(message: str) -> None:
    print(message, file=sys.stderr)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag",
        description=(
            "Ask questions over a local document corpus with citations, "
            "confidence gating, and prompt-injection defences."
        ),
    )
    parser.add_argument("--version", action="version", version=f"rag {__version__}")
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=None,
        metavar="DIR",
        help="ChromaDB directory (default: ./chroma_db, or $RAG_PERSIST_DIR)",
    )
    parser.add_argument(
        "--collection",
        default=None,
        metavar="NAME",
        help="Collection name (default: knowledge, or $RAG_COLLECTION)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser(
        "ingest", help="Index .md/.txt/.pdf files from a directory (idempotent)"
    )
    p_ingest.add_argument("path", type=Path, help="File or directory to index")
    p_ingest.add_argument(
        "--force",
        action="store_true",
        help="Re-embed every file even if its content hash is unchanged",
    )
    p_ingest.add_argument(
        "--chunk-tokens", type=int, default=None, help="Target chunk size in tokens"
    )
    p_ingest.add_argument(
        "--chunk-overlap-tokens", type=int, default=None, help="Chunk overlap in tokens"
    )

    p_ask = sub.add_parser("ask", help="Ask a single question")
    p_ask.add_argument("question", help="The question to answer")
    p_ask.add_argument(
        "--json",
        dest="json_mode",
        action="store_true",
        help="Emit a validated JSON object on stdout instead of prose",
    )
    p_ask.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve")
    p_ask.add_argument(
        "--min-score",
        type=float,
        default=None,
        metavar="F",
        help="Cosine-similarity floor for supporting evidence (0.0-1.0)",
    )
    p_ask.add_argument("--model", default=None, help="Anthropic model id")
    p_ask.add_argument(
        "--show-sources",
        action="store_true",
        help="Print the retrieved chunks and their scores (prose mode)",
    )

    p_chat = sub.add_parser("chat", help="Interactive session with conversation memory")
    p_chat.add_argument("--top-k", type=int, default=None, help="Chunks to retrieve")
    p_chat.add_argument(
        "--min-score", type=float, default=None, help="Cosine-similarity floor"
    )
    p_chat.add_argument("--model", default=None, help="Anthropic model id")

    sub.add_parser("stats", help="Show index size, indexed files, and settings")

    return parser


def _config_from_args(args: argparse.Namespace) -> RagConfig:
    return RagConfig.from_env().with_overrides(
        persist_dir=getattr(args, "persist_dir", None),
        collection_name=getattr(args, "collection", None),
        chunk_tokens=getattr(args, "chunk_tokens", None),
        chunk_overlap_tokens=getattr(args, "chunk_overlap_tokens", None),
        top_k=getattr(args, "top_k", None),
        min_score=getattr(args, "min_score", None),
        model=getattr(args, "model", None),
    )


# --------------------------------------------------------------------------
# Commands
# --------------------------------------------------------------------------


def cmd_ingest(args: argparse.Namespace, config: RagConfig) -> int:
    store = VectorStore(config)
    _err(f"Indexing {args.path} into {config.persist_dir.as_posix()} …")

    def report(result: FileResult) -> None:
        if result.status == "error":
            _err(f"  ! {result.source}: {result.detail}")
        elif result.status == "skipped":
            print(f"  = {result.source}: {result.chunks} chunks (unchanged)")
        elif result.status == "empty":
            print(f"  - {result.source}: skipped ({result.detail})")
        else:
            verb = "updated" if result.status == "updated" else "indexed"
            print(f"  + {result.source}: {result.chunks} chunks ({verb})")

    summary = ingest_path(args.path, store, config, on_file=report, force=args.force)

    if not summary.results:
        _err("No supported files found (.md, .markdown, .txt, .pdf).")
        return 1

    print(
        f"\n{len(summary.results)} file(s): "
        f"{summary.count('indexed')} indexed, "
        f"{summary.count('updated')} updated, "
        f"{summary.count('skipped')} unchanged, "
        f"{summary.count('error')} failed."
    )
    print(f"Collection now holds {store.count()} chunks.")
    return 1 if summary.count("error") else 0


def _render_prose(result: AnswerResult, show_sources: bool) -> None:
    print(result.payload.answer)
    if result.payload.citations:
        print("\nSources:")
        for citation in result.payload.citations:
            label = (
                f"{citation.source}:{citation.heading}"
                if citation.heading
                else citation.source
            )
            print(f"  - {label}  (score {citation.score:.3f})")
    print(f"\nConfidence: {result.payload.confidence}")
    if show_sources and result.used:
        print("\nRetrieved context:")
        for i, chunk in enumerate(result.used, start=1):
            preview = " ".join(chunk.text.split())[:220]
            print(f"  [{i}] {chunk.citation}  score={chunk.score:.3f}")
            print(f"      {preview}…")


def cmd_ask(args: argparse.Namespace, config: RagConfig) -> int:
    store = VectorStore(config)
    if store.count() == 0:
        _err("Index is empty. Run `rag ingest <dir>` first.")
        return 1

    agent = RagAgent(store, config)
    result = agent.answer(args.question, json_mode=args.json_mode)

    if args.json_mode:
        # ensure_ascii=False keeps non-ASCII readable rather than \uXXXX-escaped.
        # Still valid JSON; stdout is UTF-8 on every platform Python 3.11+ targets.
        print(json.dumps(result.payload.model_dump(), indent=2, ensure_ascii=False))
    else:
        _render_prose(result, args.show_sources)
        if result.gated:
            _err(
                "note: refused locally — no chunk cleared the "
                f"{config.min_score:.2f} similarity floor, so no model call was made."
            )
    return 0


def cmd_chat(args: argparse.Namespace, config: RagConfig) -> int:
    store = VectorStore(config)
    if store.count() == 0:
        _err("Index is empty. Run `rag ingest <dir>` first.")
        return 1

    agent = RagAgent(store, config)
    history: list[Turn] = []
    print(CHAT_BANNER)

    while True:
        try:
            question = input("you › ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not question:
            continue
        if question in ("/exit", "/quit"):
            return 0
        if question == "/reset":
            history.clear()
            print("(conversation memory cleared)\n")
            continue
        if question == "/stats":
            _print_stats(store, config)
            continue

        try:
            result = agent.answer(question, history=history)
        except (MissingAPIKeyError, InvalidJSONResponseError, RuntimeError) as exc:
            _err(f"error: {exc}")
            continue

        print(f"\nrag › {result.payload.answer}")
        if result.payload.citations:
            labels = ", ".join(
                f"{c.source}:{c.heading}" if c.heading else c.source
                for c in result.payload.citations
            )
            print(f"      sources: {labels}")
        print(f"      confidence: {result.payload.confidence}\n")

        history.append(
            Turn(
                user_content=result.user_content or question,
                assistant_content=result.payload.answer,
            )
        )
        history[:] = history[-config.history_turns :]


def _print_stats(store: VectorStore, config: RagConfig) -> None:
    sources = store.sources()
    print(f"Collection      : {config.collection_name}")
    print(f"Persist dir     : {config.persist_dir.as_posix()}")
    print(f"Distance metric : {store.distance_space}")
    print(f"Chunks indexed  : {store.count()}")
    print(f"Files indexed   : {len(sources)}")
    if sources:
        width = max(len(name) for name in sources)
        for name, count in sources.items():
            print(f"  {name.ljust(width)}  {count} chunks")
    print(f"Embedding model : {config.embedding_model} (local, sentence-transformers)")
    print(f"Answer model    : {config.model} (Anthropic API)")
    print(
        f"Chunking        : ~{config.chunk_tokens} tokens, "
        f"{config.chunk_overlap_tokens} overlap"
    )
    print(
        f"Retrieval       : top-{config.top_k}, "
        f"min score {config.min_score:.2f}, "
        f"context budget {config.max_context_tokens} tokens"
    )


def cmd_stats(args: argparse.Namespace, config: RagConfig) -> int:
    store = VectorStore(config)
    _print_stats(store, config)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    try:
        config = _config_from_args(args)
    except ValueError as exc:
        _err(f"error: {exc}")
        return 2

    handlers = {
        "ingest": cmd_ingest,
        "ask": cmd_ask,
        "chat": cmd_chat,
        "stats": cmd_stats,
    }
    try:
        return handlers[args.command](args, config)
    except MissingAPIKeyError as exc:
        _err(f"error: {exc}")
        return 1
    except InvalidJSONResponseError as exc:
        _err(f"error: {exc}")
        return 1
    except FileNotFoundError as exc:
        _err(f"error: {exc}")
        return 1
    except KeyboardInterrupt:  # pragma: no cover - interactive
        _err("\ninterrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
