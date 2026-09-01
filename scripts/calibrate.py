#!/usr/bin/env python3
"""Reproduce the retrieval measurements quoted in docs/design-notes.md.

Runs entirely on the local index — no API key, no network. Every number in the
design notes comes from this script, so a reader can check the claims rather
than take them on trust.

    rag ingest sample_corpus
    python scripts/calibrate.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag_agent.config import RETRIEVAL_MODES, RagConfig  # noqa: E402
from rag_agent.retrieve import VectorStore, retrieve  # noqa: E402

# Answerable from the bundled sample corpus.
ON_TOPIC = [
    "If a Meridian-3 loses its network link mid-task, how long can it keep "
    "working, and how would I detect that from the Fleet Control API?",
    "How many charging bays should we provision for a two-shift operation?",
    "Why would a robot suddenly slow down after we loaded it more heavily?",
    "How do I clear an emergency stop?",
    "What should I check first when 20% of the fleet goes offline at once?",
    "What are the API rate limits for task dispatch?",
    "Can I reuse Meridian-2 docks?",
    "What happens if I rotate an API key?",
    "When does a robot apply a firmware update?",
    "What is the rated payload?",
    "Why did my task cancellation return 409?",
    "When do I get production console access?",
]

# Plausible questions the corpus simply does not answer. The first is a
# deliberate hard negative: it shares vocabulary with the onboarding guide.
OFF_TOPIC = [
    "What is Acme Robotics' parental leave policy?",
    "How do I submit an expense report for travel?",
    "What is the recipe for sourdough bread?",
    "Who won the 2018 football world cup?",
    "How much equity do new engineering hires receive?",
    "What is the office dress code?",
    "How do I reset my payroll direct deposit?",
]

# Exact-token queries: the case dense retrieval is theoretically weakest on.
# Each pairs a query with the document filename that should answer it.
EXACT_TOKEN = [
    ("offline_seconds", "api-reference"),
    ("robot_unreachable", "api-reference"),
    ("error code 409", "api-reference"),
    ("X-Acme-Signature", "api-reference"),
    ("insufficient_role", "api-reference"),
    ("make sim-outage", "onboarding-guide"),
    ("LiFePO4", "product-specs"),
    ("Dockyard-2 bank of eight", "product-specs"),
    ("tasks:batch", "api-reference"),
    ("fleet_live_ key prefix", "api-reference"),
]


def best_support(store: VectorStore, config: RagConfig, question: str, mode: str):
    """The statistic the confidence gate actually uses.

    A question is refused only when *every* returned chunk falls below the
    floor, so the gate turns on the maximum support across the top-k — not on
    the top-ranked chunk, which under fusion need not be the best-supported one.
    """
    hits = retrieve(store, question, config, mode=mode, min_score=0.0)
    return max((hit.score for hit in hits), default=0.0)


def separation_table(store: VectorStore, config: RagConfig, floor: float) -> None:
    print(f"Confidence-gate separation (floor = {floor:.2f})\n")
    print(
        f"  {'mode':<8} {'on min':>7} {'on med':>7} {'off max':>8} {'gap':>8} "
        f"{'false acc':>10} {'false ref':>10}"
    )
    baseline: dict[str, float] = {}
    for mode in RETRIEVAL_MODES:
        on = sorted(best_support(store, config, q, mode) for q in ON_TOPIC)
        off = sorted(best_support(store, config, q, mode) for q in OFF_TOPIC)
        if mode == "vector":
            baseline = {q: best_support(store, config, q, mode) for q in ON_TOPIC}
        accepts = sum(1 for s in off if s >= floor)
        refusals = sum(1 for s in on if s < floor)
        print(
            f"  {mode:<8} {on[0]:>7.3f} {on[len(on) // 2]:>7.3f} {off[-1]:>8.3f} "
            f"{on[0] - off[-1]:>+8.3f} {accepts:>7}/{len(off)} {refusals:>7}/{len(on)}"
        )

    regressions = [
        q
        for q in ON_TOPIC
        if best_support(store, config, q, "hybrid") < baseline[q] - 1e-9
    ]
    print(
        "\n  hybrid vs vector: "
        + (
            "no question loses support"
            if not regressions
            else f"{len(regressions)} REGRESSION(S)"
        )
    )
    for question in regressions:
        print(f"    - {question}")


def exact_token_table(store: VectorStore, config: RagConfig) -> None:
    print("\n\nExact-token queries — is the right document in the top 5?\n")
    print(f"  {'query':<28} {'vector':>8} {'bm25':>8} {'hybrid':>8}")
    totals = dict.fromkeys(RETRIEVAL_MODES, 0)
    for query, expected in EXACT_TOKEN:
        row = []
        for mode in RETRIEVAL_MODES:
            hits = retrieve(store, query, config, mode=mode, top_k=5, min_score=0.0)
            rank = next(
                (i for i, hit in enumerate(hits, 1) if expected in hit.source), None
            )
            totals[mode] += rank is not None
            row.append(f"#{rank}" if rank else "—")
        print(f"  {query:<28} {row[0]:>8} {row[1]:>8} {row[2]:>8}")
    found = "  ".join(f"{m}={totals[m]}/{len(EXACT_TOKEN)}" for m in RETRIEVAL_MODES)
    print(f"\n  found in top-5:  {found}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--persist-dir", type=Path, default=None)
    args = parser.parse_args()

    config = RagConfig.from_env().with_overrides(persist_dir=args.persist_dir)
    store = VectorStore(config)

    if store.count() == 0:
        print("Index is empty. Run `rag ingest sample_corpus` first.", file=sys.stderr)
        return 1

    print(f"Corpus: {store.count()} chunks from {len(store.sources())} files\n")
    separation_table(store, config, config.min_score)
    exact_token_table(store, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
