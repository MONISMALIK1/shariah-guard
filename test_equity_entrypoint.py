import json
import os
import pytest
from equity_entrypoint import screen_equity_investment


@pytest.fixture
def log_path(tmp_path):
    return str(tmp_path / "equity_screening_log.jsonl")


def test_failing_stock_explanation_and_log(log_path):
    out = screen_equity_investment(
        ticker="RISKY",
        revenue=1_000_000,
        impermissible_income=60_000,
        total_interest_bearing_debt=400_000,
        market_cap=1_000_000,
        dividend_received=100.0,
        log_path=log_path,
    )
    assert "does not currently pass" in out.customer_explanation
    assert "RISKY" in out.customer_explanation
    assert out.board_audit_log["passed"] is False
    assert len(out.board_audit_log["reasons"]) == 2
    assert out.board_audit_log["escalated_to_board"] is False
    assert out.board_audit_log["decision_method"] == "deterministic_rule"


def test_passing_stock_with_purification_note(log_path):
    out = screen_equity_investment(
        ticker="CLEAN",
        revenue=1_000_000,
        impermissible_income=20_000,
        total_interest_bearing_debt=100_000,
        market_cap=1_000_000,
        dividend_received=500.0,
        log_path=log_path,
    )
    assert "passes Shari'ah equity screening" in out.customer_explanation
    assert "purification amount of 10.00" in out.customer_explanation
    assert out.board_audit_log["passed"] is True
    assert out.board_audit_log["purification_amount"] == 10.0


def test_passing_stock_with_zero_impermissible_income_no_purification_note(log_path):
    out = screen_equity_investment(
        ticker="PUREST",
        revenue=1_000_000,
        impermissible_income=0,
        total_interest_bearing_debt=100_000,
        market_cap=1_000_000,
        dividend_received=500.0,
        log_path=log_path,
    )
    assert "purification" not in out.customer_explanation
    assert out.board_audit_log["purification_amount"] == 0.0


def test_log_is_appended_not_overwritten(log_path):
    screen_equity_investment(
        ticker="A", revenue=1000, impermissible_income=0,
        total_interest_bearing_debt=0, market_cap=1000, log_path=log_path,
    )
    screen_equity_investment(
        ticker="B", revenue=1000, impermissible_income=0,
        total_interest_bearing_debt=0, market_cap=1000, log_path=log_path,
    )
    with open(log_path) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert lines[0]["ticker"] == "A"
    assert lines[1]["ticker"] == "B"


def test_standard_reference_present_in_every_record(log_path):
    out = screen_equity_investment(
        ticker="X", revenue=1000, impermissible_income=0,
        total_interest_bearing_debt=0, market_cap=1000, log_path=log_path,
    )
    assert "AAOIFI Shari'ah Standard No. 21" in out.board_audit_log["standard_reference"]
