import pytest
from citation_guard import validate_citations


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
