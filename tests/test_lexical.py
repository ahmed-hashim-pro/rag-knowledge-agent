"""BM25 tokenization, scoring, persistence, and rank fusion."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_agent.lexical import (
    BM25Index,
    corpus_fingerprint,
    reciprocal_rank_fusion,
    tokenize,
)

DOCS = [
    "The robot buffers its task queue and safe-parks after twelve minutes.",
    "offline_seconds counts up from the last successful telemetry report.",
    "Feed the sourdough starter equal weights of flour and water each morning.",
]
IDS = ["a", "b", "c"]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build(IDS, DOCS, corpus_fingerprint(IDS, ["h1", "h2", "h3"]))


# -- tokenization ----------------------------------------------------------


def test_code_identifiers_are_kept_whole_and_split() -> None:
    tokens = tokenize("offline_seconds")
    assert "offline_seconds" in tokens, "the literal identifier must be matchable"
    assert "offline" in tokens and "seconds" in tokens, "so must its parts"


def test_punctuation_and_case_are_normalised() -> None:
    assert tokenize("GET /robots/{robot_id}/telemetry") == [
        "get",
        "robots",
        "robot_id",
        "robot",
        "id",
        "telemetry",
    ]


def test_digits_survive_tokenization() -> None:
    """Error codes are exactly the queries dense retrieval is weakest on."""
    assert "409" in tokenize("returned a 409 error")


def test_stopwords_and_single_characters_are_dropped() -> None:
    assert tokenize("the a of I x") == []


# -- scoring ---------------------------------------------------------------


def test_exact_identifier_match_wins(index: BM25Index) -> None:
    [(top_id, score)] = index.search("offline_seconds", 1)
    assert top_id == "b"
    assert score > 0


def test_unrelated_query_scores_nothing(index: BM25Index) -> None:
    assert index.search("quantum chromodynamics", 5) == []


def test_results_are_ordered_by_score(index: BM25Index) -> None:
    hits = index.search("robot telemetry report", 3)
    assert hits == sorted(hits, key=lambda hit: -hit[1])


def test_rare_terms_outrank_common_ones() -> None:
    docs = ["robot robot robot", "robot sourdough"] + ["robot"] * 8
    idx = BM25Index.build([str(i) for i in range(10)], docs, corpus_fingerprint([], []))
    [(top_id, _)] = idx.search("sourdough", 1)
    assert top_id == "1", "IDF should favour the document with the rare term"


def test_limit_is_respected(index: BM25Index) -> None:
    assert len(index.search("robot telemetry sourdough", 2)) <= 2


def test_empty_index_searches_cleanly() -> None:
    empty = BM25Index.build([], [], corpus_fingerprint([], []))
    assert empty.search("anything", 5) == []
    assert empty.size == 0


# -- persistence -----------------------------------------------------------


def test_round_trip_preserves_results(index: BM25Index, tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    index.save(path)
    reloaded = BM25Index.load(path)

    assert reloaded is not None
    assert reloaded.fingerprint == index.fingerprint
    assert reloaded.search("offline_seconds", 3) == index.search("offline_seconds", 3)


def test_missing_index_is_a_cache_miss(tmp_path: Path) -> None:
    assert BM25Index.load(tmp_path / "absent.json") is None


def test_corrupt_index_is_a_cache_miss_not_a_crash(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    path.write_text("{not json at all", encoding="utf-8")
    assert BM25Index.load(path) is None


def test_future_format_version_is_a_cache_miss(tmp_path: Path) -> None:
    path = tmp_path / "bm25.json"
    path.write_text(json.dumps({"format": 999}), encoding="utf-8")
    assert BM25Index.load(path) is None


def test_save_is_atomic(index: BM25Index, tmp_path: Path) -> None:
    """A crash mid-write must not leave a half-written index in place."""
    path = tmp_path / "bm25.json"
    index.save(path)
    index.save(path)
    assert path.exists()
    assert not path.with_suffix(".json.tmp").exists()


# -- fingerprints ----------------------------------------------------------


def test_fingerprint_changes_when_content_changes() -> None:
    before = corpus_fingerprint(["a", "b"], ["h1", "h2"])
    assert corpus_fingerprint(["a", "b"], ["h1", "CHANGED"]) != before


def test_fingerprint_is_order_independent() -> None:
    assert corpus_fingerprint(["a", "b"], ["h1", "h2"]) == corpus_fingerprint(
        ["b", "a"], ["h2", "h1"]
    )


# -- fusion ----------------------------------------------------------------


def test_agreement_between_rankings_beats_a_single_top_hit() -> None:
    """The property that makes RRF useful — and that anchoring must offset."""
    fused = reciprocal_rank_fusion([["solo", "both"], ["both", "other"]], k=60)
    assert fused["both"] > fused["solo"]


def test_fusion_sums_reciprocal_ranks() -> None:
    fused = reciprocal_rank_fusion([["x"], ["x"]], k=60)
    assert fused["x"] == pytest.approx(2 / 61)


def test_fusion_of_nothing_is_empty() -> None:
    assert reciprocal_rank_fusion([], k=60) == {}
