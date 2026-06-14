"""
Inventory & Perishability Agent
Dynamic Pricing POC — LangGraph + Gemini

Graph nodes:
    load_csv          → check_perishable → compute_expiry → compute_loss
                      → assign_urgency   → call_llm       → build_output
                      → advance_row      → (loop or END)

Run:
    python inventory_agent.py --api-key YOUR_KEY --csv products.csv
"""

import argparse
import csv
import json
from datetime import datetime, timezone
from typing import TypedDict, Optional, List

from langgraph.graph import StateGraph, END
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage


# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are a pricing agent for a retail grocery store.
You will be given inventory data for a perishable product that is at risk of expiring unsold.
Your job is to recommend an ideal selling price that:
1. Aggressively clears units before expiry
2. Never goes below the cost_price
3. Minimises total loss compared to loss_if_no_action

Respond ONLY with a valid JSON object using exactly this schema:
{
  "suggested_action": "DISCOUNT",
  "price_modifier": <float between 0 and 1, e.g. 0.65 means 65% of current price>,
  "confidence_score": <float between 0 and 1>,
  "urgency": <"IMMEDIATE" | "HIGH" | "MEDIUM">,
  "headline": "<one line summary>",
  "detailed_reasoning": "<two to three sentence explanation>"
}
No preamble, no markdown fences, only the JSON object."""


# ── Shared state ───────────────────────────────────────────────────────────────
class AgentState(TypedDict):
    # inputs
    csv_path:   str
    api_key:    str
    # populated by load_csv_node
    rows:       List[dict]
    # current row being processed
    current_row: Optional[dict]
    # populated by check_perishable_node
    is_perishable: Optional[bool]
    # populated by compute_expiry_node
    days_to_expiry: Optional[float]
    units_at_risk:  Optional[float]
    # populated by compute_loss_node
    expiry_loss_rate:  Optional[float]
    loss_if_no_action: Optional[float]
    # populated by assign_urgency_node
    urgency: Optional[str]
    # populated by call_llm_node
    llm_response: Optional[dict]
    # accumulated across all rows
    results:   List[dict]
    # internal cursor
    row_index: int
    # populated by sort_by_urgency_node — full sorted processing queue
    urgency_queue: List[dict]


# ── Node 1: load_csv ───────────────────────────────────────────────────────────
def load_csv_node(state: AgentState) -> AgentState:
    """Reads the CSV and loads all rows into state."""
    with open(state["csv_path"], newline="") as f:
        state["rows"] = list(csv.DictReader(f))
    state["row_index"]     = 0
    state["results"]       = []
    state["urgency_queue"] = []
    print(f"[load_csv] Loaded {len(state['rows'])} row(s)")
    return state


# ── Node 1b: sort_by_urgency ───────────────────────────────────────────────────
# Urgency tier weights — lower number = processed first
URGENCY_RANK = {"IMMEDIATE": 0, "HIGH": 1, "MEDIUM": 2, "SKIP": 3}

def _precompute_urgency(row: dict) -> tuple[str, float]:
    """
    Lightweight pre-scan for a single row — no LLM, pure Python math.
    Returns (urgency_label, loss_if_no_action) for sorting purposes.
    Rows that are non-perishable or have no units at risk return ("SKIP", 0.0).
    """
    if row.get("is_perishable", "").strip().upper() != "TRUE":
        return "SKIP", 0.0

    try:
        now    = datetime.now(timezone.utc)
        expiry = datetime.fromisoformat(
            row["expiry_datetime"].replace("Z", "+00:00")
        )
        days_to_expiry = max((expiry - now).total_seconds() / 86400, 0)
        avg_daily      = float(row["avg_daily_units_sold"])
        stock          = float(row["stock_on_hand"])
        units_at_risk  = stock - (avg_daily * days_to_expiry)

        if units_at_risk <= 0:
            return "SKIP", 0.0

        buyback          = float(row["producer_buyback_rate"])
        repurposing      = float(row["repurposing_recovery_rate"])
        cost_price       = float(row["cost_price"])
        expiry_loss_rate = 1.0 - buyback - repurposing
        loss_if_no_action = round(units_at_risk * cost_price * expiry_loss_rate, 2)

        urgency = (
            "IMMEDIATE" if days_to_expiry <= 1
            else "HIGH"  if days_to_expiry <= 3
            else "MEDIUM"
        )
        return urgency, loss_if_no_action

    except (KeyError, ValueError):
        return "SKIP", 0.0


def sort_by_urgency_node(state: AgentState) -> AgentState:
    """
    Pre-scans ALL rows using pure Python math (no LLM).
    Sorts by:
      1. Urgency tier  — IMMEDIATE → HIGH → MEDIUM  (primary)
      2. loss_if_no_action descending                (tiebreaker)
    Prints the full processing queue before any LLM call is made.
    Replaces state["rows"] with the sorted order so the downstream
    row-by-row loop picks them up in priority sequence.
    """
    scored = []
    for row in state["rows"]:
        urgency, loss = _precompute_urgency(row)
        scored.append({
            "sku_id":           row.get("sku_id", "UNKNOWN"),
            "product_name":     row.get("product_name", ""),
            "urgency":          urgency,
            "loss_if_no_action": loss,
            "row":              row,
        })

    # Sort: urgency tier first (ascending rank), loss descending as tiebreaker
    scored.sort(
        key=lambda x: (URGENCY_RANK[x["urgency"]], -x["loss_if_no_action"])
    )

    # Replace rows with sorted order so the loop processes them in priority order
    state["rows"] = [s["row"] for s in scored]

    # Build the urgency queue — used only for display
    state["urgency_queue"] = [
        {k: v for k, v in s.items() if k != "row"}
        for s in scored
    ]

    # ── Print the full processing queue before any LLM call ───────────────────
    URGENCY_ICONS = {"IMMEDIATE": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "SKIP": "⚪"}
    separator = "─" * 62

    print(f"\n[sort_by_urgency] {separator}")
    print(f"[sort_by_urgency]  PROCESSING QUEUE  ({len(scored)} SKU(s) total)")
    print(f"[sort_by_urgency] {separator}")
    print(f"[sort_by_urgency]  {'#':<4} {'SKU':<10} {'URGENCY':<11} "
          f"{'LOSS IF NO ACTION':<20} PRODUCT")
    print(f"[sort_by_urgency] {separator}")

    for i, s in enumerate(scored, 1):
        icon    = URGENCY_ICONS[s["urgency"]]
        loss    = f"${s['loss_if_no_action']:.2f}" if s["urgency"] != "SKIP" else "—"
        print(
            f"[sort_by_urgency]  {i:<4} {s['sku_id']:<10} "
            f"{icon} {s['urgency']:<9} {loss:<20} {s['product_name']}"
        )

    print(f"[sort_by_urgency] {separator}")

    actionable = [s for s in scored if s["urgency"] != "SKIP"]
    skipped    = [s for s in scored if s["urgency"] == "SKIP"]
    print(f"[sort_by_urgency]  Actionable: {len(actionable)}   "
          f"Skipped (no risk): {len(skipped)}")
    print(f"[sort_by_urgency] {separator}\n")

    return state


# ── Node 2: check_perishable ───────────────────────────────────────────────────
def check_perishable_node(state: AgentState) -> AgentState:
    """
    Picks the current row and checks whether it is perishable.
    Writes True/False to is_perishable — the conditional edge
    uses this to skip non-perishable SKUs immediately.
    """
    row = state["rows"][state["row_index"]]
    state["current_row"]   = row
    state["is_perishable"] = row.get("is_perishable", "").strip().upper() == "TRUE"
    sku = row.get("sku_id", "UNKNOWN")
    print(f"[check_perishable] [{sku}] is_perishable={state['is_perishable']}")
    return state


# ── Node 3: compute_expiry ─────────────────────────────────────────────────────
def compute_expiry_node(state: AgentState) -> AgentState:
    """
    Computes days_to_expiry and units_at_risk from the current row.
    If units_at_risk <= 0 the row does not need intervention;
    the conditional edge will skip ahead to advance_row.
    """
    row = state["current_row"]
    sku = row.get("sku_id", "UNKNOWN")

    now    = datetime.now(timezone.utc)
    expiry = datetime.fromisoformat(row["expiry_datetime"].replace("Z", "+00:00"))
    days_to_expiry = max((expiry - now).total_seconds() / 86400, 0)

    avg_daily     = float(row["avg_daily_units_sold"])
    stock         = float(row["stock_on_hand"])
    units_at_risk = stock - (avg_daily * days_to_expiry)

    state["days_to_expiry"] = round(days_to_expiry, 2)
    state["units_at_risk"]  = round(units_at_risk, 2)
    print(f"[compute_expiry] [{sku}] days_to_expiry={state['days_to_expiry']}  "
          f"units_at_risk={state['units_at_risk']}")
    return state


# ── Node 4: compute_loss ───────────────────────────────────────────────────────
def compute_loss_node(state: AgentState) -> AgentState:
    """
    Computes the net expiry loss rate (after producer buyback and
    repurposing recovery) and the total dollar loss if no action is taken.
    """
    row = state["current_row"]
    sku = row.get("sku_id", "UNKNOWN")

    buyback     = float(row["producer_buyback_rate"])
    repurposing = float(row["repurposing_recovery_rate"])
    cost_price  = float(row["cost_price"])

    expiry_loss_rate  = 1.0 - buyback - repurposing
    loss_if_no_action = state["units_at_risk"] * cost_price * expiry_loss_rate

    state["expiry_loss_rate"]  = round(expiry_loss_rate, 4)
    state["loss_if_no_action"] = round(loss_if_no_action, 2)
    print(f"[compute_loss] [{sku}] expiry_loss_rate={state['expiry_loss_rate']}  "
          f"loss_if_no_action=${state['loss_if_no_action']}")
    return state


# ── Node 5: assign_urgency ─────────────────────────────────────────────────────
def assign_urgency_node(state: AgentState) -> AgentState:
    """
    Assigns an urgency label based on days_to_expiry.
    IMMEDIATE (<= 1 day), HIGH (<= 3 days), MEDIUM (> 3 days).
    """
    d = state["days_to_expiry"]
    state["urgency"] = (
        "IMMEDIATE" if d <= 1
        else "HIGH"  if d <= 3
        else "MEDIUM"
    )
    sku = state["current_row"].get("sku_id", "UNKNOWN")
    print(f"[assign_urgency] [{sku}] urgency={state['urgency']}")
    return state


# ── Node 6: call_llm ───────────────────────────────────────────────────────────
def call_llm_node(state: AgentState) -> AgentState:
    """Builds the prompt from computed metrics and calls Gemini."""
    row = state["current_row"]
    sku = row.get("sku_id", "UNKNOWN")

    prompt = f"""Product: {row['product_name']}
Category: {row['category']}
Unit: {row['unit']}
Stock on hand: {row['stock_on_hand']}
Days to expiry: {state['days_to_expiry']}
Avg daily units sold: {row['avg_daily_units_sold']}
Units sold last 24h: {row['units_sold_last_24h']}
Units at risk of expiry: {state['units_at_risk']}
Cost price (floor): ${float(row['cost_price'])}
Expiry loss rate: {state['expiry_loss_rate']}
Loss if no action taken: ${state['loss_if_no_action']}

Recommend a price modifier to clear stock before expiry."""

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=state["api_key"],
        temperature=0.2,
    )
    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=prompt)]
    response = llm.invoke(messages)
    raw = response.content.strip().strip("```json").strip("```").strip()
    state["llm_response"] = json.loads(raw)
    print(f"[call_llm] [{sku}] modifier={state['llm_response']['price_modifier']}  "
          f"confidence={state['llm_response']['confidence_score']}")
    return state


# ── Node 7: build_output ───────────────────────────────────────────────────────
def build_output_node(state: AgentState) -> AgentState:
    """Assembles the final JSON proposal and appends it to results."""
    row = state["current_row"]
    llm = state["llm_response"]

    output = {
        "agent_id":  "inventory_perishability",
        "sku_id":    row["sku_id"],
        "status":    "COMPLETED",
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "metrics_evaluated": {
            "product_name":         row["product_name"],
            "category":             row["category"],
            "unit":                 row["unit"],
            "stock_on_hand":        int(row["stock_on_hand"]),
            "days_to_expiry":       state["days_to_expiry"],
            "avg_daily_units_sold": float(row["avg_daily_units_sold"]),
            "units_sold_last_24h":  int(row["units_sold_last_24h"]),
            "units_at_risk":        state["units_at_risk"],
            "cost_price":           float(row["cost_price"]),
            "expiry_loss_rate":     state["expiry_loss_rate"],
            "loss_if_no_action":    state["loss_if_no_action"],
        },
        "proposal": {
            "suggested_action": llm["suggested_action"],
            "price_modifier":   llm["price_modifier"],
            "confidence_score": llm["confidence_score"],
            "urgency":          state["urgency"],
        },
        "justification": {
            "headline":           llm["headline"],
            "detailed_reasoning": llm["detailed_reasoning"],
        },
    }
    state["results"].append(output)
    print(f"[build_output] [{row['sku_id']}] Proposal ready")
    return state


# ── Node 8: advance_row ────────────────────────────────────────────────────────
def advance_row_node(state: AgentState) -> AgentState:
    """Increments the row cursor to move to the next SKU."""
    state["row_index"] += 1
    return state


# ── Conditional edge: is this SKU perishable? ──────────────────────────────────
def route_perishable(state: AgentState) -> str:
    return "compute_expiry" if state["is_perishable"] else "advance_row"


# ── Conditional edge: are any units at risk? ───────────────────────────────────
def route_units_at_risk(state: AgentState) -> str:
    return "compute_loss" if state["units_at_risk"] > 0 else "advance_row"


# ── Conditional edge: are there more rows? ─────────────────────────────────────
def route_more_rows(state: AgentState) -> str:
    return "check_perishable" if state["row_index"] < len(state["rows"]) else END


# ── Build graph ────────────────────────────────────────────────────────────────
def build_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("load_csv",          load_csv_node)
    graph.add_node("sort_by_urgency",   sort_by_urgency_node)   # ← new
    graph.add_node("check_perishable",  check_perishable_node)
    graph.add_node("compute_expiry",    compute_expiry_node)
    graph.add_node("compute_loss",      compute_loss_node)
    graph.add_node("assign_urgency",    assign_urgency_node)
    graph.add_node("call_llm",          call_llm_node)
    graph.add_node("build_output",      build_output_node)
    graph.add_node("advance_row",       advance_row_node)

    graph.set_entry_point("load_csv")
    graph.add_edge("load_csv",        "sort_by_urgency")        # ← new edge
    graph.add_edge("sort_by_urgency", "check_perishable")       # ← new edge

    graph.add_conditional_edges("check_perishable", route_perishable, {
        "compute_expiry": "compute_expiry",
        "advance_row":    "advance_row",
    })

    graph.add_conditional_edges("compute_expiry", route_units_at_risk, {
        "compute_loss": "compute_loss",
        "advance_row":  "advance_row",
    })

    graph.add_edge("compute_loss",   "assign_urgency")
    graph.add_edge("assign_urgency", "call_llm")
    graph.add_edge("call_llm",       "build_output")
    graph.add_edge("build_output",   "advance_row")

    graph.add_conditional_edges("advance_row", route_more_rows, {
        "check_perishable": "check_perishable",
        END:                END,
    })

    return graph.compile()


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Inventory Perishability Agent")
    parser.add_argument("--api-key", required=True, help="Google Gemini API key")
    parser.add_argument("--csv", default="products.csv", help="Path to inventory CSV")
    args = parser.parse_args()

    app = build_graph()

    initial_state: AgentState = {
        "csv_path":          args.csv,
        "api_key":           args.api_key,
        "rows":              [],
        "current_row":       None,
        "is_perishable":     None,
        "days_to_expiry":    None,
        "units_at_risk":     None,
        "expiry_loss_rate":  None,
        "loss_if_no_action": None,
        "urgency":           None,
        "llm_response":      None,
        "results":           [],
        "row_index":         0,
        "urgency_queue":     [],
    }

    final_state = app.invoke(initial_state)

    print(f"\n✅ Done — {len(final_state['results'])} proposal(s) generated\n")
    for output in final_state["results"]:
        print(json.dumps(output, indent=2))
        print()


if __name__ == "__main__":
    main()
