"""Pure-logic tests — no API key needed, no network calls."""
import pytest
from aaoifi_screening import screen_equity, IMPERMISSIBLE_INCOME_THRESHOLD, DEBT_TO_MARKET_CAP_THRESHOLD


def test_clean_company_passes():
    result = screen_equity(
        revenue=1_000_000, impermissible_income=10_000,
        total_interest_bearing_debt=200_000, market_cap=1_000_000,
    )
    assert result.passed
    assert result.reasons == []
    assert result.impermissible_income_ratio == pytest.approx(0.01)
    assert result.debt_ratio == pytest.approx(0.20)


def test_fails_on_income_ratio():
    result = screen_equity(
        revenue=1_000_000, impermissible_income=60_000,
        total_interest_bearing_debt=100_000, market_cap=1_000_000,
    )
    assert not result.passed
    assert len(result.reasons) == 1
    assert "Impermissible income" in result.reasons[0]


def test_fails_on_debt_ratio():
    result = screen_equity(
        revenue=1_000_000, impermissible_income=10_000,
        total_interest_bearing_debt=400_000, market_cap=1_000_000,
    )
    assert not result.passed
    assert "Debt-to-market-cap" in result.reasons[0]


def test_fails_on_both():
    result = screen_equity(
        revenue=1_000_000, impermissible_income=60_000,
        total_interest_bearing_debt=400_000, market_cap=1_000_000,
    )
    assert not result.passed
    assert len(result.reasons) == 2


def test_exact_threshold_is_a_fail_not_a_pass():
    # AAOIFI thresholds are "< 5%" — exactly 5% must fail, not pass, since the
    # rule is a strict inequality. This is the boundary condition most likely
    # to get silently inverted in a careless implementation.
    result = screen_equity(
        revenue=1_000_000,
        impermissible_income=1_000_000 * IMPERMISSIBLE_INCOME_THRESHOLD,
        total_interest_bearing_debt=100_000, market_cap=1_000_000,
    )
    assert not result.passed


def test_purification_applies_even_when_passed():
    result = screen_equity(
        revenue=1_000_000, impermissible_income=20_000,
        total_interest_bearing_debt=100_000, market_cap=1_000_000,
    )
    assert result.passed
    # 2% impermissible income ratio -> 2% of any dividend must be purified
    assert result.purification_amount(dividend_received=500.0) == pytest.approx(10.0)


def test_zero_impermissible_income_means_zero_purification():
    result = screen_equity(
        revenue=1_000_000, impermissible_income=0,
        total_interest_bearing_debt=100_000, market_cap=1_000_000,
    )
    assert result.purification_amount(dividend_received=1000.0) == 0.0


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        screen_equity(revenue=0, impermissible_income=0, total_interest_bearing_debt=0, market_cap=1000)
    with pytest.raises(ValueError):
        screen_equity(revenue=1000, impermissible_income=0, total_interest_bearing_debt=0, market_cap=0)
    with pytest.raises(ValueError):
        screen_equity(revenue=1000, impermissible_income=-1, total_interest_bearing_debt=0, market_cap=1000)
