"""
Deterministic AAOIFI equity screening.

This is intentionally NOT an LLM call. AAOIFI's screening thresholds are
published, numeric, and enumerable — running them through an LLM would add
hallucination risk to a calculation that's just arithmetic. Anywhere a rule
can be reduced to a number and a threshold, it belongs in code, not a prompt.
The LLM's job (see shariah_guard.py) is reserved for genuinely qualitative
questions the numbers can't answer.

Thresholds (AAOIFI Shari'ah Standard No. 21 — Financial Paper):
- Impermissible (interest/non-compliant) income must be < 5% of total revenue.
- Interest-bearing debt must be < 30% of market capitalization.
A stock can pass both screens and still require "purification" — donating the
proportional impermissible-income share of any dividend received.
"""
from dataclasses import dataclass, field

IMPERMISSIBLE_INCOME_THRESHOLD = 0.05
DEBT_TO_MARKET_CAP_THRESHOLD = 0.30


@dataclass
class ScreeningResult:
    passed: bool
    impermissible_income_ratio: float
    debt_ratio: float
    reasons: list[str] = field(default_factory=list)

    def purification_amount(self, dividend_received: float) -> float:
        """The portion of a dividend that must be purified (donated), regardless
        of whether the stock passed screening — purification applies to ANY
        impermissible income ratio > 0, pass or fail."""
        return round(dividend_received * self.impermissible_income_ratio, 2)


def screen_equity(
    revenue: float,
    impermissible_income: float,
    total_interest_bearing_debt: float,
    market_cap: float,
) -> ScreeningResult:
    """Run the two AAOIFI numeric screens. Raises ValueError on invalid inputs
    rather than silently dividing by zero or returning a misleading result."""
    if revenue <= 0:
        raise ValueError("revenue must be positive")
    if market_cap <= 0:
        raise ValueError("market_cap must be positive")
    if impermissible_income < 0 or total_interest_bearing_debt < 0:
        raise ValueError("impermissible_income and debt must be non-negative")

    income_ratio = impermissible_income / revenue
    debt_ratio = total_interest_bearing_debt / market_cap

    reasons = []
    if income_ratio >= IMPERMISSIBLE_INCOME_THRESHOLD:
        reasons.append(
            f"Impermissible income ratio {income_ratio:.2%} meets or exceeds the "
            f"{IMPERMISSIBLE_INCOME_THRESHOLD:.0%} AAOIFI threshold"
        )
    if debt_ratio >= DEBT_TO_MARKET_CAP_THRESHOLD:
        reasons.append(
            f"Debt-to-market-cap ratio {debt_ratio:.2%} meets or exceeds the "
            f"{DEBT_TO_MARKET_CAP_THRESHOLD:.0%} AAOIFI threshold"
        )

    return ScreeningResult(
        passed=len(reasons) == 0,
        impermissible_income_ratio=income_ratio,
        debt_ratio=debt_ratio,
        reasons=reasons,
    )
