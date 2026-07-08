"""
Separate entry point: "can we invest in this stock" (AAOIFI equity screening).

This is deliberately NOT wired into shariah_guard.py's retrieve -> LLM ->
ground-check graph. That graph exists to manage the risk of an LLM's
qualitative judgment on product structure ("is this Mudarabah compliant").
Equity screening isn't a judgment call — it's arithmetic against a published
threshold (see aaoifi_screening.py). Forcing it through the citation-grounding
check would be building a safeguard against a failure mode (hallucinated
citations) that literally cannot occur here, since no LLM output is involved.

Still emits the same two-audience shape as the qualitative pipeline — a plain
customer explanation and a full board-facing audit log — because that
requirement ("every decision emitted twice") applies to the decision, not to
how the decision was produced.
"""
import datetime
import json
from dataclasses import dataclass

from aaoifi_screening import screen_equity

AAOIFI_STANDARD_REFERENCE = "AAOIFI Shari'ah Standard No. 21 (Financial Paper) — equity screening thresholds"


@dataclass
class EquityScreeningOutput:
    customer_explanation: str
    board_audit_log: dict


def screen_equity_investment(
    *,
    ticker: str,
    revenue: float,
    impermissible_income: float,
    total_interest_bearing_debt: float,
    market_cap: float,
    dividend_received: float = 0.0,
    log_path: str = "equity_screening_log.jsonl",
) -> EquityScreeningOutput:
    result = screen_equity(
        revenue=revenue,
        impermissible_income=impermissible_income,
        total_interest_bearing_debt=total_interest_bearing_debt,
        market_cap=market_cap,
    )
    purification = result.purification_amount(dividend_received) if dividend_received else 0.0

    if result.passed:
        customer_explanation = f"{ticker} passes Shari'ah equity screening."
        if purification > 0:
            customer_explanation += (
                f" A small purification amount of {purification:.2f} applies to your "
                f"dividend, since a small portion of the company's income isn't "
                f"Shari'ah-compliant — this should be donated, not spent."
            )
    else:
        customer_explanation = (
            f"{ticker} does not currently pass Shari'ah equity screening: "
            + "; ".join(r.rstrip(".") for r in result.reasons) + "."
        )

    audit_log = {
        "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "ticker": ticker,
        "standard_reference": AAOIFI_STANDARD_REFERENCE,
        "passed": result.passed,
        "impermissible_income_ratio": round(result.impermissible_income_ratio, 4),
        "debt_ratio": round(result.debt_ratio, 4),
        "reasons": result.reasons,
        "dividend_received": dividend_received,
        "purification_amount": purification,
        # No AI output is involved in this decision, so there is nothing for a
        # citation-grounding check to catch and no ambiguity for a board to
        # resolve — the audit trail IS the justification.
        "escalated_to_board": False,
        "decision_method": "deterministic_rule",
    }

    with open(log_path, "a") as f:
        f.write(json.dumps(audit_log) + "\n")

    return EquityScreeningOutput(customer_explanation=customer_explanation, board_audit_log=audit_log)


if __name__ == "__main__":
    out = screen_equity_investment(
        ticker="EXMP",
        revenue=1_000_000,
        impermissible_income=60_000,
        total_interest_bearing_debt=100_000,
        market_cap=1_000_000,
        dividend_received=250.0,
    )
    print("Customer sees:", out.customer_explanation)
    print("Board record: ", out.board_audit_log)

    out2 = screen_equity_investment(
        ticker="CLEAN",
        revenue=1_000_000,
        impermissible_income=10_000,
        total_interest_bearing_debt=200_000,
        market_cap=1_000_000,
        dividend_received=250.0,
    )
    print("\nCustomer sees:", out2.customer_explanation)
    print("Board record: ", out2.board_audit_log)
