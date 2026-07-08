import pytest
from citation_guard import validate_citations, extract_citations_from_text


def test_all_citations_valid():
    result = validate_citations(cited_sources=[1, 2], num_passages_shown=3)
    assert result.is_grounded
    assert result.valid_citations == [1, 2]
    assert result.hallucinated_citations == []


def test_hallucinated_citation_detected():
    # Model cites [4] but only 3 passages were ever shown to it — fabricated.
    result = validate_citations(cited_sources=[1, 4], num_passages_shown=3)
    assert not result.is_grounded
    assert result.valid_citations == [1]
    assert result.hallucinated_citations == [4]


def test_zero_citation_index_is_hallucinated():
    # Citations are 1-indexed; 0 or negative never corresponds to a real passage.
    result = validate_citations(cited_sources=[0, 1], num_passages_shown=3)
    assert not result.is_grounded
    assert result.hallucinated_citations == [0]


def test_no_citations_at_all_is_not_grounded():
    # A decision with zero citations isn't "grounded" — it's unsupported.
    result = validate_citations(cited_sources=[], num_passages_shown=5)
    assert not result.is_grounded


def test_all_citations_hallucinated():
    result = validate_citations(cited_sources=[9, 10], num_passages_shown=3)
    assert not result.is_grounded
    assert result.valid_citations == []
    assert result.hallucinated_citations == [9, 10]


def test_zero_passages_shown_means_any_citation_is_hallucinated():
    result = validate_citations(cited_sources=[1], num_passages_shown=0)
    assert not result.is_grounded
    assert result.hallucinated_citations == [1]


def test_negative_num_passages_raises():
    with pytest.raises(ValueError):
        validate_citations(cited_sources=[1], num_passages_shown=-1)


# ── extract_citations_from_text ─────────────────────────────────────────

def test_extract_single_citations():
    text = "This is confirmed in [1] and further supported by [3]."
    assert extract_citations_from_text(text) == [1, 3]


def test_extract_grouped_citation_bracket():
    text = "Multiple passages [1, 2] establish this requirement."
    assert extract_citations_from_text(text) == [1, 2]


def test_extract_dedupes_repeated_citations():
    text = "As stated in [1], and again in [1], this holds. See also [2]."
    assert extract_citations_from_text(text) == [1, 2]


def test_extract_returns_sorted_regardless_of_order_in_text():
    text = "First [3], then [1], then [2]."
    assert extract_citations_from_text(text) == [1, 2, 3]


def test_extract_no_citations_present():
    text = "This text has no bracket citations at all."
    assert extract_citations_from_text(text) == []


def test_extract_ignores_non_numeric_brackets():
    # Bracketed non-numeric content (e.g. a stray footnote marker) shouldn't
    # be mistaken for a citation.
    text = "See the appendix [see below] and passage [2]."
    assert extract_citations_from_text(text) == [2]
