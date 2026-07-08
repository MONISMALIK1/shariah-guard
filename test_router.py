"""
Tests for the pure branching logic only — route_from_classification and
log_routing_decision need zero API access. classify_query() and route()
(which calls the real model, and route() also calls the real shariah_guard
graph) are exercised live, not here — same split as the rest of this project.
"""
import json
import pytest
from router import (
    QueryCategory,
    route_from_classification,
    log_routing_decision,
    OUT_OF_SCOPE_MESSAGE,
    EQUITY_SCREENING_REDIRECT_MESSAGE,
)


def test_out_of_scope_routes_correctly_and_never_touches_the_graph():
    classification = QueryCategory(category="out_of_scope", reasoning="This is a personal Zakat question.")
    result = route_from_classification("How is Zakat calculated on savings?", classification)
    assert result["routed_to"] == "out_of_scope"
    assert result["customer_explanation"] == OUT_OF_SCOPE_MESSAGE
    assert result["classifier_reasoning"] == "This is a personal Zakat question."


def test_equity_screening_query_redirects_with_guidance():
    classification = QueryCategory(category="equity_screening", reasoning="Asks whether a stock is investable.")
    result = route_from_classification("Can I invest in this company's stock?", classification)
    assert result["routed_to"] == "equity_screening_redirect"
    assert result["customer_explanation"] == EQUITY_SCREENING_REDIRECT_MESSAGE


def test_institutional_compliance_is_flagged_for_forwarding_not_answered_here():
    classification = QueryCategory(category="institutional_compliance", reasoning="Asks about a Mudarabah structure.")
    result = route_from_classification("Is this Mudarabah structure compliant?", classification)
    assert result["routed_to"] == "institutional_compliance"
    # No customer_explanation yet at this layer — that only exists after the
    # real graph runs, which route_from_classification deliberately doesn't do.
    assert "customer_explanation" not in result


def test_log_routing_decision_appends_with_timestamp(tmp_path):
    log_path = str(tmp_path / "router_log.jsonl")
    result = {"query": "test", "routed_to": "out_of_scope", "classifier_reasoning": "x", "customer_explanation": "y"}
    log_routing_decision(result, log_path=log_path)
    log_routing_decision(result, log_path=log_path)

    with open(log_path) as f:
        lines = [json.loads(line) for line in f]
    assert len(lines) == 2
    assert "timestamp" in lines[0]
    assert lines[0]["routed_to"] == "out_of_scope"
