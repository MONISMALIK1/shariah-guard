"""
The single front door for free-text queries.

Without this, a caller has to already know whether their question is an
institutional compliance question or something entirely out of scope (a
personal Zakat question, a general fiqh question unrelated to banking) — and
an out-of-scope question was burning a full retrieve + structured-LLM-ruling
cycle just to land on an unhelpful "requires_review" anyway, as we saw live
with the Zakat query.

This adds one cheap classification call in front of that expensive pipeline.
The branching logic itself (route_from_classification) is pure and testable
without any API call — only classify_query() and route() need a live model.
"""
import datetime
import json
import os
from typing import Literal

from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

OUT_OF_SCOPE_MESSAGE = (
    "This falls outside our institutional Shari'ah compliance scope (bank "
    "product/transaction structures and equity screening). For personal "
    "worship obligations or general fiqh questions, please consult a "
    "qualified scholar directly — this system's knowledge base does not "
    "cover that and shouldn't be trusted to answer it."
)

EQUITY_SCREENING_REDIRECT_MESSAGE = (
    "This looks like an equity/investment screening question. That path "
    "needs specific financial figures (revenue, impermissible income, "
    "interest-bearing debt, market cap) rather than a free-text query — "
    "please provide those figures directly."
)


class QueryCategory(BaseModel):
    category: Literal["institutional_compliance", "equity_screening", "out_of_scope"] = Field(
        description="institutional_compliance: a bank product/transaction structure "
        "question (e.g. is this Mudarabah/loan/fee structure compliant). "
        "equity_screening: a question about whether a stock/company is investable "
        "(needs financial figures we don't have from text alone). "
        "out_of_scope: personal worship obligations (Zakat, prayer, etc.) or any "
        "general fiqh question unrelated to institutional banking."
    )
    reasoning: str = Field(description="One sentence explaining the classification")


CLASSIFIER_SYSTEM_PROMPT = """Classify the query into exactly one category. Do not
answer the query itself — only classify it. Be decisive; when a query is clearly
about personal religious obligations (Zakat, prayer, fasting, etc.) rather than a
bank's products or an investment decision, it is out_of_scope."""

_classifier_chain = None


def _get_classifier_chain():
    # Constructed lazily, on first real use, rather than at module import —
    # ChatGoogleGenerativeAI validates its API key eagerly at construction
    # time, which would otherwise make this module unimportable (and its pure
    # branching logic untestable) in any environment without GEMINI_API_KEY set.
    global _classifier_chain
    if _classifier_chain is None:
        prompt = ChatPromptTemplate.from_messages([
            ("system", CLASSIFIER_SYSTEM_PROMPT),
            ("human", "{query}"),
        ])
        model = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=os.environ["GEMINI_API_KEY"],
            max_tokens=300,  # classification only — deliberately cheap vs. the ~4096 ruling call
        ).with_structured_output(QueryCategory)
        _classifier_chain = prompt | model
    return _classifier_chain


def classify_query(query: str) -> QueryCategory:
    return _get_classifier_chain().invoke({"query": query})


def route_from_classification(query: str, classification: QueryCategory) -> dict:
    """Pure branching logic, no API call — the part that's actually testable."""
    if classification.category == "out_of_scope":
        return {
            "query": query,
            "routed_to": "out_of_scope",
            "classifier_reasoning": classification.reasoning,
            "customer_explanation": OUT_OF_SCOPE_MESSAGE,
        }

    if classification.category == "equity_screening":
        return {
            "query": query,
            "routed_to": "equity_screening_redirect",
            "classifier_reasoning": classification.reasoning,
            "customer_explanation": EQUITY_SCREENING_REDIRECT_MESSAGE,
        }

    return {
        "query": query,
        "routed_to": "institutional_compliance",
        "classifier_reasoning": classification.reasoning,
    }


def log_routing_decision(result: dict, log_path: str = "router_log.jsonl") -> None:
    entry = {"timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(), **result}
    with open(log_path, "a") as f:
        f.write(json.dumps(entry) + "\n")


def route(query: str) -> dict:
    """The actual front door. Only reaches the expensive shariah_guard graph
    when the cheap classifier says this is genuinely in scope for it."""
    classification = classify_query(query)
    result = route_from_classification(query, classification)

    if result["routed_to"] == "institutional_compliance":
        import shariah_guard
        result.update(shariah_guard.run(query))

    log_routing_decision(result)
    return result


if __name__ == "__main__":
    import sys

    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else input("Enter a query: ")
    result = route(query)
    print(f"\nRouted to: {result['routed_to']}")
    print(f"Classifier reasoning: {result['classifier_reasoning']}")
    print(f"Customer sees: {result['customer_explanation']}")
