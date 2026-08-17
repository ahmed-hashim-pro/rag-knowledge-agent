"""Ingestion: idempotency, change detection, loaders, and path hygiene."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_agent.config import RagConfig
from rag_agent.ingest import (
    file_digest,
    ingest_path,
    iter_source_files,
    load_document,
    relative_source,
)


def test_ingest_indexes_every_supported_file(
    corpus: Path, store, config: RagConfig
) -> None:
    summary = ingest_path(corpus, store, config)

    assert summary.count("indexed") == 3
    assert summary.total_chunks == store.count()
    assert set(store.sources()) == {
        "corpus/battery.md",
        "corpus/network.md",
        "corpus/baking.md",
    }


def test_reingesting_unchanged_corpus_is_a_no_op(
    corpus: Path, store, config: RagConfig
) -> None:
    first = ingest_path(corpus, store, config)
    chunks_after_first = store.count()

    second = ingest_path(corpus, store, config)

    assert second.count("skipped") == 3
    assert second.count("indexed") == 0
    assert second.count("updated") == 0
    assert store.count() == chunks_after_first
    assert second.total_chunks == first.total_chunks


def test_changed_file_is_reindexed_and_old_chunks_are_replaced(
    corpus: Path, store, config: RagConfig
) -> None:
    ingest_path(corpus, store, config)
    before = store.sources()["corpus/battery.md"]

    (corpus / "battery.md").write_text(
        "# Battery\n\n## Charging\n\nRewritten content about charging cycles.\n\n"
        "## Replacement\n\nPacks are swapped after 2000 cycles.\n",
        encoding="utf-8",
    )
    summary = ingest_path(corpus, store, config)

    assert summary.count("updated") == 1
    assert summary.count("skipped") == 2

    payload = store.collection.get(
        where={"source": "corpus/battery.md"}, include=["documents", "metadatas"]
    )
    documents = payload["documents"]
    assert all("45 minutes" not in doc for doc in documents), "stale chunks removed"
    assert any("2000 cycles" in doc for doc in documents)
    assert len(documents) == store.sources()["corpus/battery.md"]
    assert before  # the file was indexed before the edit


def test_force_reindexes_even_when_unchanged(
    corpus: Path, store, config: RagConfig
) -> None:
    ingest_path(corpus, store, config)
    summary = ingest_path(corpus, store, config, force=True)

    assert summary.count("indexed") + summary.count("updated") == 3
    assert summary.count("skipped") == 0


def test_deleting_a_file_leaves_the_others_intact(
    corpus: Path, store, config: RagConfig
) -> None:
    ingest_path(corpus, store, config)
    (corpus / "baking.md").unlink()

    summary = ingest_path(corpus, store, config)

    assert summary.count("skipped") == 2
    # A vanished file is not pruned automatically; its chunks remain queryable.
    assert "corpus/baking.md" in store.sources()


def test_chunk_metadata_is_complete_and_never_none(
    corpus: Path, store, config: RagConfig
) -> None:
    ingest_path(corpus, store, config)
    payload = store.collection.get(include=["metadatas"])

    for metadata in payload["metadatas"]:
        assert set(metadata) == {
            "source",
            "heading",
            "chunk_index",
            "file_hash",
            "chars",
        }
        assert all(value is not None for value in metadata.values())
        assert isinstance(metadata["heading"], str)


def test_file_digest_tracks_content_not_mtime(tmp_path: Path) -> None:
    path = tmp_path / "a.md"
    path.write_text("hello", encoding="utf-8")
    first = file_digest(path)

    path.write_text("hello", encoding="utf-8")
    assert file_digest(path) == first

    path.write_text("goodbye", encoding="utf-8")
    assert file_digest(path) != first


def test_iter_source_files_filters_and_skips_hidden(tmp_path: Path) -> None:
    (tmp_path / "keep.md").write_text("x", encoding="utf-8")
    (tmp_path / "keep.txt").write_text("x", encoding="utf-8")
    (tmp_path / "skip.png").write_bytes(b"\x89PNG")
    hidden = tmp_path / ".git"
    hidden.mkdir()
    (hidden / "config.md").write_text("x", encoding="utf-8")

    found = {p.name for p in iter_source_files(tmp_path)}
    assert found == {"keep.md", "keep.txt"}


def test_relative_source_never_leaks_an_absolute_path(tmp_path: Path) -> None:
    outside = tmp_path / "elsewhere" / "notes.md"
    outside.parent.mkdir()
    outside.write_text("x", encoding="utf-8")

    label = relative_source(outside, root=tmp_path / "corpus")

    assert not label.startswith("/")
    assert ".." not in label
    assert label.endswith("notes.md")


def test_txt_files_are_loaded_as_plain_text(tmp_path: Path) -> None:
    path = tmp_path / "notes.txt"
    path.write_text("plain body\n", encoding="utf-8")
    assert load_document(path) == "plain body\n"


def test_unsupported_extension_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "image.png"
    path.write_bytes(b"\x89PNG")
    with pytest.raises(ValueError, match="Unsupported file type"):
        load_document(path)


def _write_minimal_pdf(path: Path, text: str = "Meridian offline autonomy") -> None:
    """Write a real one-page PDF with extractable text.

    Built with pypdf rather than checked in as a binary fixture, so the PDF
    path is exercised against the same library that reads it.
    """
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)

    stream = DecodedStreamObject()
    stream.set_data(f"BT /F1 12 Tf 40 200 Td ({text}) Tj ET".encode())
    page[NameObject("/Contents")] = writer._add_object(stream)

    font = DictionaryObject()
    font.update(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    fonts = DictionaryObject()
    fonts[NameObject("/F1")] = writer._add_object(font)
    resources = DictionaryObject()
    resources[NameObject("/Font")] = fonts
    page[NameObject("/Resources")] = resources

    with path.open("wb") as handle:
        writer.write(handle)


def test_pdf_pages_become_headed_sections(tmp_path: Path, store, config) -> None:
    pdf = tmp_path / "handbook.pdf"
    _write_minimal_pdf(pdf)

    text = load_document(pdf)
    assert text.startswith("## Page 1"), "each PDF page gets a citable heading"
    assert "Meridian offline autonomy" in text

    summary = ingest_path(pdf, store, config)
    assert summary.count("indexed") == 1

    payload = store.collection.get(include=["metadatas"])
    assert payload["metadatas"][0]["heading"] == "Page 1"


def test_unreadable_file_is_reported_not_raised(
    tmp_path: Path, store, config: RagConfig
) -> None:
    good = tmp_path / "good.md"
    good.write_text("# Good\n\nbody\n", encoding="utf-8")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not really a pdf")

    summary = ingest_path(tmp_path, store, config)

    assert summary.count("indexed") == 1
    assert summary.count("error") == 1
    assert any(
        r.status == "error" and r.source.endswith("bad.pdf") for r in summary.results
    )


def test_missing_path_raises(tmp_path: Path, store, config: RagConfig) -> None:
    with pytest.raises(FileNotFoundError):
        ingest_path(tmp_path / "nope", store, config)
