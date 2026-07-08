"""
Every decision is emitted twice, for two different audiences with two
different needs:

- Customer explanation: plain language, no jargon, no citation numbers —
  a customer doesn't need to see "[1][3]", they need to know what happened
  and why in one or two sentences.
- Board audit log: everything — the full reasoning, every citation (valid
  AND hallucinated, so a reviewer can see exactly what the grounding check
  caught), the escalation trail if this went to a human, and a timestamp.

This module only ASSEMBLES the two outputs from already-computed pieces
(decision, grounding result, optional escalation outcome). It does not call
an LLM — the two text fields (customer_explanation, board_reasoning) are
produced by the model itself in shariah_guard.py, since writing in two
registers is a language task, not a formatting task. What belongs here is
the non-negotiable, code-enforced structure every record must have.
"""
import datetime
from dataclasses import dataclass, field
from typing import Optional

from citation_guard import GroundingResult


@dataclass
class DualOutput:
    customer_explanation: str
    board_audit_log: dict = field(default_factory=dict)


def build_dual_output(
    *,
    query: str,
    decision: str,
    confidence: str,
    customer_explanation: str,
    board_reasoning: str,
    cited_sources: list[int],
    grounding: GroundingResult,
    escalated: bool,
    board_reviewer_note: Optional[str] = None,
) -> DualOutput:
    if not grounding.is_grounded and not escalated:
        raise ValueError(
            "A non-grounded decision (hallucinated citation present) must be "
            "escalated — it cannot be auto-emitted as a customer-facing decision."
        )

    audit_log = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "query": query,
        "decision": decision,
        "confidence": confidence,
        "board_reasoning": board_reasoning,
        "cited_sources_claimed": cited_sources,
        "cited_sources_valid": grounding.valid_citations,
        "cited_sources_hallucinated": grounding.hallucinated_citations,
        "is_grounded": grounding.is_grounded,
        "escalated_to_board": escalated,
        "board_reviewer_note": board_reviewer_note,
    }

    return DualOutput(customer_explanation=customer_explanation, board_audit_log=audit_log)
