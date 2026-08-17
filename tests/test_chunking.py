"""Chunking: structure awareness, size targeting, and overlap."""

from __future__ import annotations

import pytest

from rag_agent.config import estimate_tokens
from rag_agent.ingest import chunk_markdown

SAMPLE = """\
Preamble text that appears before any heading at all.

# Top Level

Intro paragraph under the top level heading.

## Alpha

Alpha body paragraph one.

Alpha body paragraph two.

## Beta

Beta body paragraph.

### Beta Child

Nested body paragraph.
"""


def test_heading_path_is_tracked_hierarchically() -> None:
    chunks = chunk_markdown(SAMPLE, chunk_tokens=500, overlap_tokens=50)
    headings = [chunk.heading for chunk in chunks]

    assert "" in headings, "text before the first heading keeps an empty heading"
    assert "Top Level" in headings
    assert "Top Level > Alpha" in headings
    assert "Top Level > Beta" in headings
    assert "Top Level > Beta > Beta Child" in headings


def test_chunks_never_span_two_headings() -> None:
    chunks = chunk_markdown(SAMPLE, chunk_tokens=500, overlap_tokens=50)
    alpha = next(c for c in chunks if c.heading == "Top Level > Alpha")
    assert "Beta body paragraph" not in alpha.text
    assert "Alpha body paragraph one." in alpha.text
    assert "Alpha body paragraph two." in alpha.text


def test_chunk_indices_are_sequential_and_unique() -> None:
    chunks = chunk_markdown(SAMPLE, chunk_tokens=60, overlap_tokens=10)
    indices = [chunk.index for chunk in chunks]
    assert indices == list(range(len(chunks)))


def test_long_section_is_split_near_the_token_target() -> None:
    body = "\n\n".join(
        f"Sentence block number {i} with filler words." for i in range(60)
    )
    text = f"# Doc\n\n## Long\n\n{body}\n"

    chunks = chunk_markdown(text, chunk_tokens=100, overlap_tokens=20)

    assert len(chunks) > 1
    for chunk in chunks:
        # Overlap is prepended after the size check, so allow it in the budget.
        assert estimate_tokens(chunk.text) <= 100 + 20 + 10


def test_consecutive_chunks_overlap() -> None:
    body = "\n\n".join(
        f"Paragraph {i} carries distinct filler content." for i in range(40)
    )
    text = f"# Doc\n\n## Long\n\n{body}\n"

    chunks = chunk_markdown(text, chunk_tokens=100, overlap_tokens=30)
    assert len(chunks) >= 2

    first_tail = chunks[0].text.split("\n\n")[-1]
    assert first_tail in chunks[1].text, "the tail of a chunk repeats in the next one"


def test_zero_overlap_produces_disjoint_chunks() -> None:
    body = "\n\n".join(f"Paragraph {i} unique marker." for i in range(30))
    text = f"# Doc\n\n## Long\n\n{body}\n"

    chunks = chunk_markdown(text, chunk_tokens=60, overlap_tokens=0)
    seen: set[str] = set()
    for chunk in chunks:
        for paragraph in chunk.text.split("\n\n"):
            assert paragraph not in seen
            seen.add(paragraph)


def test_headings_inside_fenced_code_are_not_headings() -> None:
    text = (
        "# Real Heading\n\n"
        "Body text.\n\n"
        "```bash\n"
        "# this is a shell comment, not a heading\n"
        "make bootstrap\n"
        "```\n\n"
        "Trailing body.\n"
    )
    chunks = chunk_markdown(text, chunk_tokens=500, overlap_tokens=50)

    assert {chunk.heading for chunk in chunks} == {"Real Heading"}
    combined = "\n".join(chunk.text for chunk in chunks)
    assert "make bootstrap" in combined


def test_fenced_code_block_is_kept_intact() -> None:
    text = '# Doc\n\n```json\n{\n  "a": 1,\n\n  "b": 2\n}\n```\n'
    chunks = chunk_markdown(text, chunk_tokens=500, overlap_tokens=50)

    joined = "\n\n".join(chunk.text for chunk in chunks)
    assert '"a": 1' in joined and '"b": 2' in joined
    # The blank line inside the fence must not have split the block.
    assert sum(chunk.text.count("```") for chunk in chunks) == 2


def test_single_oversized_paragraph_is_hard_split() -> None:
    giant = "word " * 4000
    chunks = chunk_markdown(f"# Doc\n\n{giant}\n", chunk_tokens=100, overlap_tokens=10)

    assert len(chunks) > 1
    assert all(chunk.heading == "Doc" for chunk in chunks)


def test_empty_document_yields_no_chunks() -> None:
    assert chunk_markdown("   \n\n  \n", chunk_tokens=100, overlap_tokens=10) == []


def test_overlap_must_be_smaller_than_chunk_size() -> None:
    with pytest.raises(ValueError, match="smaller than chunk_tokens"):
        chunk_markdown("# X\n\nbody\n", chunk_tokens=50, overlap_tokens=50)
