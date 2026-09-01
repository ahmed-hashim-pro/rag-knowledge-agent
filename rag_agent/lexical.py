"""BM25 lexical retrieval, as a counterweight to dense vector search.

Dense retrieval matches meaning and is blind to exact tokens: a query for
``offline_seconds`` or ``409`` competes on topical similarity with every passage
that merely discusses telemetry or errors. BM25 matches the literal term and
weights it by how rare it is, which is precisely the case embeddings lose.

Implemented directly rather than pulled from a library so the index can be
persisted alongside the Chroma collection and the tokenizer can be tuned for a
corpus full of code identifiers.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

INDEX_FILENAME = "bm25_index.json"
INDEX_FORMAT_VERSION = 1

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# Deliberately short. BM25's IDF already discounts common words; this list only
# removes terms so frequent they bloat the postings without ever discriminating.
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "our",
        "so",
        "than",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "was",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "you",
        "your",
    ]
)


def tokenize(text: str) -> list[str]:
    """Lowercase and split, keeping code identifiers whole *and* split.

    ``offline_seconds`` yields ``offline_seconds``, ``offline``, ``seconds``, so
    a query matches whether it names the identifier or describes it.
    """
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        if len(token) < 2 or token in _STOPWORDS:
            continue
        tokens.append(token)
        if "_" in token:
            tokens.extend(
                part
                for part in token.split("_")
                if len(part) >= 2 and part not in _STOPWORDS
            )
    return tokens


def corpus_fingerprint(ids: Sequence[str], file_hashes: Sequence[str]) -> str:
    """Identify the exact corpus state an index was built from."""
    digest = hashlib.sha256()
    for chunk_id, file_hash in sorted(zip(ids, file_hashes, strict=True)):
        digest.update(f"{chunk_id}\t{file_hash}\n".encode())
    return digest.hexdigest()


@dataclass
class BM25Index:
    """An in-memory BM25 index with a JSON on-disk form."""

    ids: list[str]
    doc_lengths: list[int]
    postings: dict[str, list[tuple[int, int]]]
    average_length: float
    fingerprint: str
    k1: float = DEFAULT_K1
    b: float = DEFAULT_B

    @property
    def size(self) -> int:
        return len(self.ids)

    @classmethod
    def build(
        cls,
        ids: Sequence[str],
        documents: Sequence[str],
        fingerprint: str,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> BM25Index:
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        doc_lengths: list[int] = []

        for position, document in enumerate(documents):
            counts: dict[str, int] = defaultdict(int)
            for token in tokenize(document):
                counts[token] += 1
            doc_lengths.append(sum(counts.values()))
            for token, frequency in counts.items():
                postings[token].append((position, frequency))

        total = sum(doc_lengths)
        return cls(
            ids=list(ids),
            doc_lengths=doc_lengths,
            postings=dict(postings),
            average_length=(total / len(doc_lengths)) if doc_lengths else 0.0,
            fingerprint=fingerprint,
            k1=k1,
            b=b,
        )

    def search(self, query: str, limit: int) -> list[tuple[str, float]]:
        """Return up to ``limit`` ``(chunk_id, bm25_score)`` pairs, best first."""
        if not self.ids or limit <= 0:
            return []

        scores: dict[int, float] = defaultdict(float)
        total_docs = len(self.ids)

        for term in set(tokenize(query)):
            postings = self.postings.get(term)
            if not postings:
                continue
            containing = len(postings)
            idf = math.log(1.0 + (total_docs - containing + 0.5) / (containing + 0.5))
            for position, frequency in postings:
                length_norm = (
                    1.0
                    - self.b
                    + self.b
                    * (
                        self.doc_lengths[position] / self.average_length
                        if self.average_length
                        else 0.0
                    )
                )
                denominator = frequency + self.k1 * length_norm
                scores[position] += idf * frequency * (self.k1 + 1.0) / denominator

        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(self.ids[position], score) for position, score in ranked[:limit]]

    # -- persistence -------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "format": INDEX_FORMAT_VERSION,
            "fingerprint": self.fingerprint,
            "k1": self.k1,
            "b": self.b,
            "average_length": self.average_length,
            "ids": self.ids,
            "doc_lengths": self.doc_lengths,
            "postings": {
                term: [list(pair) for pair in entries]
                for term, entries in self.postings.items()
            },
        }

    @classmethod
    def from_dict(cls, payload: dict) -> BM25Index:
        if payload.get("format") != INDEX_FORMAT_VERSION:
            raise ValueError(
                f"unsupported BM25 index format: {payload.get('format')!r}"
            )
        return cls(
            ids=list(payload["ids"]),
            doc_lengths=list(payload["doc_lengths"]),
            postings={
                term: [(int(p), int(f)) for p, f in entries]
                for term, entries in payload["postings"].items()
            },
            average_length=float(payload["average_length"]),
            fingerprint=str(payload["fingerprint"]),
            k1=float(payload.get("k1", DEFAULT_K1)),
            b=float(payload.get("b", DEFAULT_B)),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(self.to_dict()), encoding="utf-8")
        temporary.replace(path)

    @classmethod
    def load(cls, path: Path) -> BM25Index | None:
        """Read an index, or return ``None`` if it is absent or unreadable.

        A corrupt or stale-format index is a cache miss, not a failure: the
        caller rebuilds from the collection.
        """
        if not path.exists():
            return None
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError, KeyError, TypeError):
            return None


def reciprocal_rank_fusion(
    rankings: Iterable[Sequence[str]], k: int
) -> dict[str, float]:
    """Combine ranked id lists by reciprocal rank: ``sum(1 / (k + rank))``.

    Ranks rather than scores, because BM25 and cosine similarity are not on a
    common scale and normalising them against each other requires corpus-specific
    constants that RRF avoids needing.
    """
    fused: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for position, identifier in enumerate(ranking, start=1):
            fused[identifier] += 1.0 / (k + position)
    return dict(fused)
