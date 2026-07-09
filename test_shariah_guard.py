"""
Tests for the graph's actual wiring — routing, escalation, and the real
interrupt/resume cycle — with zero network calls. retrieve_node and
decide_node are exercised for real, but their expensive dependencies
(_get_retriever, _get_decision_chain) are swapped for fakes, so the graph
itself (ground_check's routing logic, the conditional edge, the
shariah_board interrupt, finalize) runs unmodified and genuinely gets tested.

This requires GEMINI_API_KEY to be unset-safe at import time (fixed by
making model/retriever construction lazy) — none of these tests need a
real key or a real network call.
"""
import json
import uuid

import pytest
import shariah_guard
from shariah_guard import RulingDecision


class FakeDoc:
    def __init__(self, content):
        self.page_content = content


class FakeRetriever:
    def __init__(self, docs):
        self._docs = docs

    def invoke(self, query):
        return self._docs


class FakeChain:
    def __init__(self, decision: RulingDecision):
        self._decision = decision

    def invoke(self, inputs):
        return self._decision


def _run_with_fakes(monkeypatch, tmp_path, decision: RulingDecision, num_docs: int = 3):
    """Wires the real compiled graph (shariah_guard.app) to fake retrieval
    and a fake canned decision, and runs it in a throwaway cwd so the
    board_audit_log.jsonl write in finalize_node doesn't touch the repo."""
    monkeypatch.setattr(shariah_guard, "_get_retriever", lambda: FakeRetriever(
        [FakeDoc(f"passage {i + 1} content") for i in range(num_docs)]
    ))
    monkeypatch.setattr(shariah_guard, "_get_decision_chain", lambda: FakeChain(decision))
    monkeypatch.chdir(tmp_path)

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    shariah_guard.app.invoke({"query": "test query"}, config)
    return config


def test_grounded_high_confidence_decision_auto_finalizes_without_escalation(monkeypatch, tmp_path):
    decision = RulingDecision(
        decision="compliant", confidence="high", cited_sources=[1, 2],
        customer_explanation="Yes, this is fine.", board_reasoning="Passages [1][2] confirm this.",
    )
    config = _run_with_fakes(monkeypatch, tmp_path, decision)

    state = shariah_guard.app.get_state(config)
    assert state.next == ()  # not paused — reached END
    assert state.values["escalated"] is False
    assert state.values["output"]["decision"] == "compliant"
    assert state.values["output"]["is_grounded"] is True


def test_requires_review_pauses_at_shariah_board_then_resumes_on_update(monkeypatch, tmp_path):
    decision = RulingDecision(
        decision="requires_review", confidence="low", cited_sources=[],
        customer_explanation="We need to look into this further.",
        board_reasoning="The context doesn't clearly answer this.",
    )
    config = _run_with_fakes(monkeypatch, tmp_path, decision)

    state = shariah_guard.app.get_state(config)
    assert state.next == ("shariah_board",)  # genuinely paused, not finished
    assert "output" not in state.values or not state.values.get("output")

    # Resume, same as run() does
    shariah_guard.app.update_state(config, {"board_reviewer_note": "Approved by board."})
    shariah_guard.app.invoke(None, config)

    final_state = shariah_guard.app.get_state(config)
    assert final_state.next == ()  # now actually finished
    assert final_state.values["output"]["escalated_to_board"] is True
    assert final_state.values["output"]["board_reviewer_note"] == "Approved by board."


def test_low_confidence_escalates_even_when_decision_is_compliant(monkeypatch, tmp_path):
    # The exact fix shipped for the "confidence is measured but never used"
    # gap — proven here through the real graph, not just ground_check_node
    # called in isolation.
    decision = RulingDecision(
        decision="compliant", confidence="low", cited_sources=[1],
        customer_explanation="This appears fine.", board_reasoning="Passage [1] suggests this.",
    )
    config = _run_with_fakes(monkeypatch, tmp_path, decision)

    state = shariah_guard.app.get_state(config)
    assert state.next == ("shariah_board",)  # escalated despite decision="compliant"


def test_hallucinated_citation_forces_escalation_even_with_high_confidence(monkeypatch, tmp_path):
    # Only 3 passages are retrieved (num_docs=3), but the model cites [7] —
    # a fabricated source. High stated confidence must not override this.
    decision = RulingDecision(
        decision="compliant", confidence="high", cited_sources=[1, 7],
        customer_explanation="This is fine.", board_reasoning="Passages [1][7] confirm this.",
    )
    config = _run_with_fakes(monkeypatch, tmp_path, decision, num_docs=3)

    state = shariah_guard.app.get_state(config)
    assert state.next == ("shariah_board",)
    assert state.values["hallucinated_citations"] == [7]


def test_finalize_writes_board_audit_log(monkeypatch, tmp_path):
    decision = RulingDecision(
        decision="non_compliant", confidence="high", cited_sources=[1],
        customer_explanation="This is not permissible.", board_reasoning="Passage [1] prohibits this.",
    )
    _run_with_fakes(monkeypatch, tmp_path, decision)

    log_file = tmp_path / "board_audit_log.jsonl"
    assert log_file.exists()
    entry = json.loads(log_file.read_text().strip())
    assert entry["decision"] == "non_compliant"
    assert entry["escalated_to_board"] is False
