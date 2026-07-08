import pytest
from citation_guard import GroundingResult
from dual_output import build_dual_output


def test_grounded_non_escalated_decision_builds_normally():
    grounding = GroundingResult(is_grounded=True, valid_citations=[1, 2], hallucinated_citations=[])
    out = build_dual_output(
        query="Is a Mudarabah profit-sharing investment compliant?",
        decision="compliant",
        confidence="high",
        customer_explanation="Yes — this structure is compliant.",
        board_reasoning="Passages [1][2] confirm profit-sharing without a guaranteed fixed return.",
        cited_sources=[1, 2],
        grounding=grounding,
        escalated=False,
    )
    assert out.customer_explanation == "Yes — this structure is compliant."
    assert out.board_audit_log["is_grounded"] is True
    assert out.board_audit_log["cited_sources_hallucinated"] == []
    assert out.board_audit_log["escalated_to_board"] is False
    assert "timestamp" in out.board_audit_log


def test_ungrounded_decision_must_be_escalated():
    # A hallucinated citation with escalated=False must be rejected outright —
    # this is the enforcement point for "the AI can only cite what's real."
    grounding = GroundingResult(is_grounded=False, valid_citations=[1], hallucinated_citations=[7])
    with pytest.raises(ValueError, match="must be escalated"):
        build_dual_output(
            query="...",
            decision="compliant",
            confidence="high",
            customer_explanation="...",
            board_reasoning="...",
            cited_sources=[1, 7],
            grounding=grounding,
            escalated=False,
        )


def test_ungrounded_decision_escalated_is_allowed_and_records_hallucination():
    grounding = GroundingResult(is_grounded=False, valid_citations=[1], hallucinated_citations=[7])
    out = build_dual_output(
        query="...",
        decision="requires_review",
        confidence="low",
        customer_explanation="We're reviewing this with our Shari'ah board.",
        board_reasoning="Model cited passage 7 which was never retrieved.",
        cited_sources=[1, 7],
        grounding=grounding,
        escalated=True,
        board_reviewer_note="Confirmed: citation 7 does not exist in this query's context.",
    )
    assert out.board_audit_log["cited_sources_hallucinated"] == [7]
    assert out.board_audit_log["escalated_to_board"] is True
    assert out.board_audit_log["board_reviewer_note"] == "Confirmed: citation 7 does not exist in this query's context."


def test_board_reviewer_note_defaults_to_none_when_not_escalated():
    grounding = GroundingResult(is_grounded=True, valid_citations=[1], hallucinated_citations=[])
    out = build_dual_output(
        query="...", decision="compliant", confidence="high",
        customer_explanation="...", board_reasoning="...",
        cited_sources=[1], grounding=grounding, escalated=False,
    )
    assert out.board_audit_log["board_reviewer_note"] is None
