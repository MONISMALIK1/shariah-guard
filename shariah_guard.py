"""
Shariah Guard — the layer between an AI-native bank's models and its customers.

Pipeline: retrieve -> LLM ruling (with both a customer explanation and a full
board-facing reasoning, in one structured call) -> grounded-citation check
(code, not prompting) -> route: auto-emit if grounded and decisive, escalate
to the Shari'ah board if the citation check fails OR the model itself said
requires_review -> dual output, always.

Reuses the knowledge base already built in ../sharia-law-rag/chroma_db and
the Gemini-backed decision pattern proven in decision_graph.py.
"""
import os
import json
import uuid
from typing import TypedDict, Literal, Optional

from pydantic import BaseModel, Field
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

from citation_guard import validate_citations
from dual_output import build_dual_output

SHARED_KNOWLEDGE_BASE = os.path.join(
    os.path.dirname(__file__), "..", "sharia-law-rag", "chroma_db"
)


# ── Schema — the model produces BOTH registers in one structured call ──────

class RulingDecision(BaseModel):
    decision: Literal["compliant", "non_compliant", "requires_review"] = Field(
        description="Whether the query complies with the retrieved governance standard"
    )
    confidence: Literal["high", "medium", "low"] = Field(
        description="Confidence given the available retrieved context"
    )
    cited_sources: list[int] = Field(
        description="Numbered context passages ([1], [2], ...) that were actually used"
    )
    customer_explanation: str = Field(
        description="One or two plain-language sentences for the customer. No jargon, "
        "no citation numbers, no internal reasoning — just the outcome and why, simply."
    )
    board_reasoning: str = Field(
        description="Full technical reasoning for a Shari'ah board reviewer, citing "
        "passage numbers explicitly."
    )


class GuardState(TypedDict):
    query: str
    context: str
    num_passages: int
    decision: str
    confidence: str
    cited_sources: list[int]
    customer_explanation: str
    board_reasoning: str
    is_grounded: bool
    hallucinated_citations: list[int]
    escalated: bool
    board_reviewer_note: Optional[str]
    output: dict


# ── Shared resources ─────────────────────────────────────────────────────

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = Chroma(persist_directory=SHARED_KNOWLEDGE_BASE, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 6, "fetch_k": 20})

SYSTEM_PROMPT = """You are a Shari'ah compliance ruling assistant for an AI-native Islamic bank.
Evaluate the query strictly against the retrieved context below.

Rules:
- Base every claim ONLY on the retrieved context — never outside knowledge.
- Cite the numbered passages ([1], [2], ...) you actually relied on. Never cite a
  number that wasn't shown to you.
- If the context does not clearly and specifically answer the query, return
  "requires_review" with low or medium confidence — do not guess.
- Write customer_explanation in plain language a non-expert can understand in
  one read, with zero citation numbers or jargon.
- Write board_reasoning as a full technical justification a Shari'ah board
  member would need to sign off on this decision.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", "Retrieved context:\n{context}\n\nQuery:\n{query}"),
])

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.environ["GEMINI_API_KEY"],
    max_tokens=4096,
).with_structured_output(RulingDecision)


def format_with_sources(docs) -> str:
    return "\n\n".join(f"[{i + 1}] {d.page_content}" for i, d in enumerate(docs))


# ── Nodes ────────────────────────────────────────────────────────────────

def retrieve_node(state: GuardState) -> dict:
    docs = retriever.invoke(state["query"])
    return {"context": format_with_sources(docs), "num_passages": len(docs)}


def decide_node(state: GuardState) -> dict:
    chain = prompt | model
    result = chain.invoke({"context": state["context"], "query": state["query"]})
    return {
        "decision": result.decision,
        "confidence": result.confidence,
        "cited_sources": result.cited_sources,
        "customer_explanation": result.customer_explanation,
        "board_reasoning": result.board_reasoning,
    }


def ground_check_node(state: GuardState) -> dict:
    grounding = validate_citations(state["cited_sources"], state["num_passages"])
    # Escalate if the model itself asked for review, OR if the grounding check
    # catches a fabricated citation regardless of how confident the model was —
    # a hallucinated citation overrides the model's own stated confidence.
    escalate = state["decision"] == "requires_review" or not grounding.is_grounded
    return {
        "is_grounded": grounding.is_grounded,
        "hallucinated_citations": grounding.hallucinated_citations,
        "escalated": escalate,
    }


def route_after_grounding(state: GuardState) -> str:
    return "board" if state["escalated"] else "finalize"


def shariah_board_node(state: GuardState) -> dict:
    # Execution reaches here only after interrupt resume — board_reviewer_note
    # is expected to already be injected into state via update_state().
    return {}


def finalize_node(state: GuardState) -> dict:
    from citation_guard import GroundingResult

    grounding = GroundingResult(
        is_grounded=state["is_grounded"],
        valid_citations=[c for c in state["cited_sources"] if c not in state["hallucinated_citations"]],
        hallucinated_citations=state["hallucinated_citations"],
    )
    result = build_dual_output(
        query=state["query"],
        decision=state["decision"],
        confidence=state["confidence"],
        customer_explanation=state["customer_explanation"],
        board_reasoning=state["board_reasoning"],
        cited_sources=state["cited_sources"],
        grounding=grounding,
        escalated=state["escalated"],
        board_reviewer_note=state.get("board_reviewer_note"),
    )
    with open("board_audit_log.jsonl", "a") as f:
        f.write(json.dumps(result.board_audit_log) + "\n")
    return {"output": {"customer_explanation": result.customer_explanation, **result.board_audit_log}}


# ── Assemble ─────────────────────────────────────────────────────────────

graph = StateGraph(GuardState)
graph.add_node("retrieve", retrieve_node)
graph.add_node("decide", decide_node)
graph.add_node("ground_check", ground_check_node)
graph.add_node("shariah_board", shariah_board_node)
graph.add_node("finalize", finalize_node)

graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "decide")
graph.add_edge("decide", "ground_check")
graph.add_conditional_edges("ground_check", route_after_grounding, {"board": "shariah_board", "finalize": "finalize"})
graph.add_edge("shariah_board", "finalize")
graph.add_edge("finalize", END)

app = graph.compile(checkpointer=MemorySaver(), interrupt_before=["shariah_board"])


# ── Demo runner ──────────────────────────────────────────────────────────

def run(query: str) -> dict:
    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    app.invoke({"query": query}, config)

    state = app.get_state(config)
    if state.next:
        print(f"\n--- ESCALATED TO SHARI'AH BOARD ---")
        print(f"Query: {query}")
        if state.values["hallucinated_citations"]:
            print(f"Reason: grounding check FAILED — hallucinated citations {state.values['hallucinated_citations']}")
        else:
            print(f"Reason: model itself requested review (confidence: {state.values['confidence']})")
        print(f"Board reasoning draft: {state.values['board_reasoning']}")
        note = input("Board reviewer note: ")
        app.update_state(config, {"board_reviewer_note": note})
        app.invoke(None, config)

    final = app.get_state(config).values["output"]
    print(f"\nQuery: {query}")
    print(f"Customer sees: {final['customer_explanation']}")
    print(f"Board record:  decision={final['decision']} grounded={final['is_grounded']} escalated={final['escalated_to_board']}")
    return final


if __name__ == "__main__":
    run("Is a Mudarabah profit-and-loss sharing investment between a bank and a small business compliant?")
    run("A fixed 5% annual interest personal loan.")
