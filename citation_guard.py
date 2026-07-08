"""
The grounded-citation rule.

An LLM asked to "cite your sources" will sometimes cite a source number that
was never actually retrieved — a fabricated citation that reads as confident
and verifiable but points at nothing. For a system whose entire value
proposition is "explainable, grounded decisioning," a hallucinated citation
is worse than no citation: it actively misleads a Shari'ah board reviewer
into trusting a claim that has no backing.

The fix is not "prompt it harder to be accurate" — it's a mechanical,
non-negotiable check: the only valid citation IDs for a given decision are
the indices of passages that were actually retrieved and shown to the model
for that specific query. Anything outside that range is provably fabricated,
full stop, regardless of how plausible it sounds.
"""
from dataclasses import dataclass


@dataclass
class GroundingResult:
    is_grounded: bool
    valid_citations: list[int]
    hallucinated_citations: list[int]


def validate_citations(cited_sources: list[int], num_passages_shown: int) -> GroundingResult:
    """
    cited_sources: the source indices the model claimed to rely on (1-indexed,
        matching the [1], [2], ... numbering shown in the retrieved context).
    num_passages_shown: how many passages were actually retrieved and included
        in the prompt for this query — the registry of what COULD be cited.
    """
    if num_passages_shown < 0:
        raise ValueError("num_passages_shown cannot be negative")

    valid = [c for c in cited_sources if 1 <= c <= num_passages_shown]
    hallucinated = [c for c in cited_sources if c not in valid]

    return GroundingResult(
        is_grounded=len(hallucinated) == 0 and len(cited_sources) > 0,
        valid_citations=valid,
        hallucinated_citations=hallucinated,
    )
