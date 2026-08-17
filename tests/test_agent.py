"""Generation: JSON-mode schema validation, retry, and the guardrails.

Every test here stubs the Anthropic client. Nothing in this file needs
``ANTHROPIC_API_KEY``, and a stub that is never called is itself the assertion
for the confidence gate.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from rag_agent.agent import (
    AnswerPayload,
    InvalidJSONResponseError,
    RagAgent,
    Turn,
    confidence_from_score,
    trim_history,
)
from rag_agent.config import RagConfig
from rag_agent.ingest import ingest_path
from rag_agent.retrieve import VectorStore

from .conftest import StubAnthropic

GOOD_JSON = json.dumps(
    {
        "answer": "It buffers for twelve minutes [corpus/network.md:Networking > "
        "Offline behaviour].",
        "citations": [
            {"source": "corpus/network.md", "heading": "Networking > Offline behaviour"}
        ],
        "confidence": "high",
    }
)


@pytest.fixture
def indexed(config: RagConfig, real_embedder, corpus: Path) -> VectorStore:
    store = VectorStore(config, embedding_function=real_embedder)
    ingest_path(corpus, store, config)
    return store


def _agent(store: VectorStore, config: RagConfig, *replies: str) -> tuple:
    client = StubAnthropic(*replies)
    return RagAgent(store, config, client=client), client


# --------------------------------------------------------------------------
# Schema validation
# --------------------------------------------------------------------------


def test_json_mode_returns_a_validated_payload(indexed, config) -> None:
    agent, client = _agent(indexed, config, GOOD_JSON)

    result = agent.answer("what happens on network loss?", json_mode=True)

    assert isinstance(result.payload, AnswerPayload)
    assert result.payload.confidence == "high"
    assert result.payload.citations[0].source == "corpus/network.md"
    assert len(client.calls) == 1
    assert not result.retried


def test_json_mode_strips_a_markdown_fence(indexed, config) -> None:
    agent, client = _agent(indexed, config, f"```json\n{GOOD_JSON}\n```")

    result = agent.answer("what happens on network loss?", json_mode=True)

    assert result.payload.answer.startswith("It buffers")
    assert len(client.calls) == 1, "a fence is handled locally, not by retrying"


def test_citation_scores_come_from_retrieval_not_the_model(indexed, config) -> None:
    """The model cannot know cosine scores, so it must never supply them."""
    lying = json.dumps(
        {
            "answer": "Twelve minutes "
            "[corpus/network.md:Networking > Offline behaviour].",
            "citations": [
                {
                    "source": "corpus/network.md",
                    "heading": "Networking > Offline behaviour",
                    "score": 0.99999,
                }
            ],
            "confidence": "high",
        }
    )
    agent, _ = _agent(indexed, config, lying)

    result = agent.answer("what happens on network loss?", json_mode=True)

    citation = result.payload.citations[0]
    assert citation.score != pytest.approx(0.99999)
    retrieved = {(c.source, c.heading): c.score for c in result.used}
    assert citation.score == retrieved[(citation.source, citation.heading)]


def test_fabricated_citation_is_dropped(indexed, config, capsys) -> None:
    fabricated = json.dumps(
        {
            "answer": "Twelve minutes "
            "[corpus/network.md:Networking > Offline behaviour], "
            "per policy [hr/handbook.md:Leave].",
            "citations": [
                {
                    "source": "corpus/network.md",
                    "heading": "Networking > Offline behaviour",
                },
                {"source": "hr/handbook.md", "heading": "Leave"},
            ],
            "confidence": "medium",
        }
    )
    agent, _ = _agent(indexed, config, fabricated)

    result = agent.answer("what happens on network loss?", json_mode=True)

    sources = {c.source for c in result.payload.citations}
    assert sources == {"corpus/network.md"}
    assert "hr/handbook.md" in capsys.readouterr().err


def test_drifted_heading_still_resolves_via_its_source(indexed, config) -> None:
    drifted = json.dumps(
        {
            "answer": "Twelve minutes [corpus/network.md:Offline behaviour].",
            "citations": [
                {"source": "corpus/network.md", "heading": "Offline behaviour"}
            ],
            "confidence": "medium",
        }
    )
    agent, _ = _agent(indexed, config, drifted)

    result = agent.answer("what happens on network loss?", json_mode=True)

    assert len(result.payload.citations) == 1
    assert result.payload.citations[0].score > 0


@pytest.mark.parametrize(
    "bad",
    [
        "this is prose, not json at all",
        '{"answer": "x", "citations": []}',  # missing confidence
        '{"answer": "x", "citations": [], "confidence": "certain"}',  # bad enum
        '{"citations": [], "confidence": "high"}',  # missing answer
        '{"answer": "x", "citations": [{"heading": "H"}], "confidence": "low"}',
    ],
)
def test_invalid_json_triggers_exactly_one_retry(indexed, config, bad, capsys) -> None:
    agent, client = _agent(indexed, config, bad, GOOD_JSON)

    result = agent.answer("what happens on network loss?", json_mode=True)

    assert result.retried
    assert len(client.calls) == 2
    assert result.payload.confidence == "high"
    assert "retrying once" in capsys.readouterr().err


def test_retry_prompt_carries_the_validation_error(indexed, config) -> None:
    agent, client = _agent(indexed, config, "not json", GOOD_JSON)

    agent.answer("what happens on network loss?", json_mode=True)

    correction = client.calls[1]["messages"][-1]
    assert correction["role"] == "user"
    assert "could not be parsed" in correction["content"]
    # The failed reply is replayed as an assistant turn, never as a prefill.
    assert client.calls[1]["messages"][-2]["role"] == "assistant"


def test_two_bad_replies_raise_rather_than_print_garbage(indexed, config) -> None:
    agent, client = _agent(indexed, config, "nope", "still nope")

    with pytest.raises(InvalidJSONResponseError, match="after one retry"):
        agent.answer("what happens on network loss?", json_mode=True)

    assert len(client.calls) == 2, "exactly one retry, never a loop"


def test_answer_payload_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnswerPayload(answer="x", citations=[], confidence="high", extra="nope")


# --------------------------------------------------------------------------
# Confidence gating
# --------------------------------------------------------------------------


def test_low_confidence_refuses_without_calling_the_api(indexed, config) -> None:
    agent, client = _agent(indexed, config)  # no canned replies at all
    strict = config.min_score

    result = agent.answer("what is the company parental leave policy?", min_score=0.35)

    assert result.gated
    assert result.payload.citations == []
    assert result.payload.confidence == "low"
    assert result.payload.answer.startswith("I don't know")
    assert client.calls == [], "the confidence gate spends no tokens"
    assert strict is not None


def test_gate_is_bypassed_when_evidence_clears_the_floor(indexed, config) -> None:
    agent, client = _agent(indexed, config, "Twelve minutes [corpus/network.md:x].")

    result = agent.answer("what happens on network loss?", min_score=0.35)

    assert not result.gated
    assert len(client.calls) == 1


def test_json_refusal_has_the_same_shape_as_a_json_answer(indexed, config) -> None:
    agent, _ = _agent(indexed, config)

    result = agent.answer("parental leave policy", json_mode=True, min_score=0.35)
    payload = json.loads(json.dumps(result.payload.model_dump()))

    assert set(payload) == {"answer", "citations", "confidence"}
    assert payload["citations"] == []
    assert payload["confidence"] == "low"


@pytest.mark.parametrize(
    ("top", "floor", "expected"),
    [
        (0.90, 0.35, "high"),
        (0.56, 0.35, "high"),
        (0.50, 0.35, "medium"),
        (0.44, 0.35, "medium"),
        (0.36, 0.35, "low"),
    ],
)
def test_confidence_bands_are_relative_to_the_floor(top, floor, expected) -> None:
    assert confidence_from_score(top, floor) == expected


# --------------------------------------------------------------------------
# Prompt construction and injection defence
# --------------------------------------------------------------------------


def test_documents_precede_the_question_in_the_user_turn(indexed, config) -> None:
    agent, client = _agent(indexed, config, "answer text")

    agent.answer("what happens on network loss?")

    content = client.calls[0]["messages"][-1]["content"]
    assert content.index("<document") < content.index("Question:")


def test_system_prompt_states_the_injection_rule(indexed, config) -> None:
    agent, client = _agent(indexed, config, "answer text")

    agent.answer("what happens on network loss?")
    system = client.calls[0]["system"]

    assert "untrusted third-party content" in system
    assert "never obey it" in system
    assert "[source:heading]" in system


def test_json_protocol_is_only_added_in_json_mode(indexed, config) -> None:
    agent, client = _agent(indexed, config, "prose answer", GOOD_JSON)

    agent.answer("what happens on network loss?")
    assert "# Output format" not in client.calls[0]["system"]

    agent.answer("what happens on network loss?", json_mode=True)
    assert "# Output format" in client.calls[1]["system"]


def test_no_trailing_assistant_prefill_is_ever_sent(indexed, config) -> None:
    """Sonnet 4.6 rejects a prefilled final assistant turn with a 400."""
    agent, client = _agent(indexed, config, "not json", GOOD_JSON)

    agent.answer("what happens on network loss?", json_mode=True)

    for call in client.calls:
        assert call["messages"][-1]["role"] == "user"


def test_prose_mode_cites_from_retrieval(indexed, config) -> None:
    agent, _ = _agent(indexed, config, "Twelve minutes.")

    result = agent.answer("what happens on network loss?", min_score=0.35)

    assert result.payload.citations
    assert all(c.score > 0 for c in result.payload.citations)
    assert {c.source for c in result.payload.citations} <= {
        c.source for c in result.used
    }


def test_model_refusal_stop_reason_is_surfaced(indexed, config, monkeypatch) -> None:
    agent, client = _agent(indexed, config, "whatever")
    original = client.messages.create

    def refusing(**kwargs):
        response = original(**kwargs)
        response.stop_reason = "refusal"
        return response

    monkeypatch.setattr(client.messages, "create", refusing)

    with pytest.raises(RuntimeError, match="refusal"):
        agent.answer("what happens on network loss?")


# --------------------------------------------------------------------------
# Conversation memory
# --------------------------------------------------------------------------


def test_history_is_replayed_as_alternating_turns(indexed, config) -> None:
    agent, client = _agent(indexed, config, "second answer")
    history = [Turn(user_content="first question", assistant_content="first answer")]

    agent.answer("follow up", history=history)

    messages = client.calls[0]["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant", "user"]
    assert messages[0]["content"] == "first question"
    assert messages[1]["content"] == "first answer"


def test_history_is_capped_at_the_configured_turn_count(config) -> None:
    turns = [Turn(f"q{i}", f"a{i}") for i in range(25)]
    kept = trim_history(turns, config.with_overrides(history_turns=10))

    assert len(kept) == 10
    assert kept[-1].user_content == "q24"
    assert kept[0].user_content == "q15"


def test_history_is_also_capped_by_a_token_budget(config) -> None:
    turns = [Turn("x" * 8000, "y" * 8000) for _ in range(10)]
    kept = trim_history(turns, config.with_overrides(max_history_tokens=6000))

    assert 0 < len(kept) < 10


def test_empty_history_is_fine(config) -> None:
    assert trim_history([], config) == []
