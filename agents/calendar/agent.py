"""
Calendar & Festivity Agent — v2
================================
Rebuilt as a LangGraph StateGraph (matching the Inventory Agent architecture).

Graph nodes:
    load_catalog
        │
    check_next_sku ──(done)──► END
        │
    scan_festivals
        │
    check_holidays
        │
    compute_lift
        │
    assign_urgency
        │
    call_llm
        │
    build_output
        │
    advance_row ──────────────► (loop back to check_next_sku)

Run:
    python calendar_agent.py
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import date, datetime, timedelta , timezone
from pathlib import Path
from typing import Any, Optional, TypedDict

import pandas as pd
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph

# ---------------------------------------------------------------------------
# 0.  Environment & paths
# ---------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "")
if not GEMINI_API_KEY:
    sys.exit("ERROR: GEMINI_API_KEY not set. Add it to your .env file.")

BASE_DIR            = Path(__file__).parent
CSV_PATH            = BASE_DIR / "data" / "calendar_data.csv"
FESTIVAL_JSON_PATH  = BASE_DIR / "data" / "festival_calendar.json"
LOOKAHEAD_DAYS      = 21
PROX_WINDOW_DAYS    = 7

# ---------------------------------------------------------------------------
# 1.  Mock festival calendar — auto-created on first run
# ---------------------------------------------------------------------------
MOCK_FESTIVAL_CALENDAR: list[dict] = [
    {"name": "Eid al-Adha",            "date_hint": "2026-06-06", "month": None, "day": None, "base_lift": 1.75, "categories": ["meat","sweets","dairy","bakery","spices"],           "region": "global"},
    {"name": "Eid al-Fitr",            "date_hint": "2026-03-31", "month": None, "day": None, "base_lift": 1.60, "categories": ["sweets","meat","bakery","beverages"],                 "region": "global"},
    {"name": "Christmas",              "date_hint": None,         "month": 12,   "day": 25,   "base_lift": 1.50, "categories": ["meat","dairy","bakery","beverages","snacks"],         "region": "global"},
    {"name": "Thanksgiving",           "date_hint": None,         "month": 11,   "day": 27,   "base_lift": 1.45, "categories": ["meat","dairy","bakery","produce","beverages"],        "region": "US"},
    {"name": "Diwali",                 "date_hint": "2026-11-08", "month": None, "day": None, "base_lift": 1.40, "categories": ["sweets","snacks","beverages","dairy"],                "region": "global"},
    {"name": "Easter Sunday",          "date_hint": "2026-04-05", "month": 4,    "day": 5,    "base_lift": 1.30, "categories": ["dairy","bakery","produce","meat"],                   "region": "global"},
    {"name": "Ramadan Start",          "date_hint": "2026-03-01", "month": None, "day": None, "base_lift": 1.55, "categories": ["meat","dairy","sweets","bakery","spices"],            "region": "global"},
    {"name": "New Year's Day",         "date_hint": None,         "month": 1,    "day": 1,    "base_lift": 1.20, "categories": ["beverages","snacks","dairy","meat"],                  "region": "global"},
    {"name": "Independence Day (US)",  "date_hint": None,         "month": 7,    "day": 4,    "base_lift": 1.25, "categories": ["meat","beverages","snacks","produce"],                "region": "US"},
    {"name": "Hanukkah",               "date_hint": "2026-12-05", "month": None, "day": None, "base_lift": 1.20, "categories": ["dairy","sweets","bakery"],                           "region": "global"},
]


def bootstrap_festival_json() -> None:
    """Ensures the production calendar exists, otherwise stops execution."""
    FESTIVAL_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not FESTIVAL_JSON_PATH.exists():
        sys.exit(
            f"\n❌ ERROR: {FESTIVAL_JSON_PATH} is missing!\n"
            "Please create this file and paste the fully updated JSON array "
            "(the one containing the 'surcharge_exempt' keys) into it before running."
        )


def _load_festival_calendar() -> list[dict]:
    with open(FESTIVAL_JSON_PATH) as fh:
        return json.load(fh)


def _festival_date_for_year(festival: dict, year: int) -> date | None:
    if festival.get("date_hint"):
        try:
            return date.fromisoformat(festival["date_hint"]).replace(year=year)
        except ValueError:
            pass
    if festival.get("month") and festival.get("day"):
        try:
            return date(year, festival["month"], festival["day"])
        except ValueError:
            return None
    return None


# ---------------------------------------------------------------------------
# 2.  AgentState — one field per node's output
# ---------------------------------------------------------------------------
class AgentState(TypedDict):
    # input
    csv_path:           str
    today_str:          str             # YYYY-MM-DD reference date
    # catalog
    rows:               list[dict]
    row_index:          int
    current_row:        Optional[dict]
    # festival scan
    surcharge_exempt_triggered: bool
    festival_name:      Optional[str]
    days_to_event:      int
    base_lift:          float
    categories_affected: list[str]
    # holiday check
    public_holiday:     Optional[str]
    holiday_days_away:  int
    # lift computation
    demand_lift_factor: float
    # urgency
    urgency:            Optional[str]
    # LLM
    llm_response:       Optional[dict]
    # output
    results:            list[dict]


# ---------------------------------------------------------------------------
# 3.  Node 1 — load_catalog
# ---------------------------------------------------------------------------
def load_catalog_node(state: AgentState) -> AgentState:
    """Reads the product CSV and initialises the row cursor."""
    df = pd.read_csv(state["csv_path"])
    state["rows"]      = df.to_dict(orient="records")
    state["row_index"] = 0
    state["results"]   = []
    print(f"[load_catalog] Loaded {len(state['rows'])} SKU(s) from {state['csv_path']}")
    return state


# ---------------------------------------------------------------------------
# 4.  Node 2 — check_next_sku
#     Sets current_row; routing edge decides whether to process or end.
# ---------------------------------------------------------------------------
def check_next_sku_node(state: AgentState) -> AgentState:
    """Picks the row at row_index into current_row."""
    state["current_row"] = state["rows"][state["row_index"]]
    row = state["current_row"]
    print(f"[check_next_sku] Processing SKU {row['sku_id']} — {row['product_name']}")
    return state


# ---------------------------------------------------------------------------
# 5.  Node 3 — scan_festivals
#     Finds the nearest relevant festival in the next 21 days.
# ---------------------------------------------------------------------------
def scan_festivals_node(state: AgentState) -> AgentState:
    """
    Scans the festival calendar for the next 21 days.
    Matches on the current SKU's category.
    Writes: festival_name, days_to_event, base_lift, categories_affected.
    """
    today    = date.fromisoformat(state["today_str"])
    category = str(state["current_row"].get("category", "")).lower()
    festivals = _load_festival_calendar()

    best_festival: dict | None = None
    best_days_to: int = LOOKAHEAD_DAYS + 1

    for fest in festivals:
        for yr in [today.year, today.year + 1]:
            fest_date = _festival_date_for_year(fest, yr)
            if fest_date is None:
                continue
            days_to = (fest_date - today).days
            if 0 <= days_to <= LOOKAHEAD_DAYS:
                if category in [c.lower() for c in fest["categories"]]:
                    if days_to < best_days_to:
                        best_festival = fest
                        best_days_to  = days_to
            

    if best_festival:
        is_exempt = category in [c.lower() for c in best_festival.get("surcharge_exempt", [])]
        
        state["festival_name"]              = best_festival["name"]
        state["days_to_event"]              = best_days_to
        state["categories_affected"]        = best_festival["categories"]
        state["surcharge_exempt_triggered"] = is_exempt
        # Force lift to 1.0 if exempt, otherwise use the base_lift
        state["base_lift"]                  = 1.0 if is_exempt else best_festival["base_lift"]
        
        status_msg = "EXEMPT" if is_exempt else "AFFECTED"
        print(f"[scan_festivals] Found: {best_festival['name']} in {best_days_to} day(s) -> {status_msg}")
    else:
        state["festival_name"]              = None
        state["days_to_event"]              = -1
        state["base_lift"]                  = 1.0
        state["categories_affected"]        = []
        state["surcharge_exempt_triggered"] = False
        print(f"[scan_festivals] No relevant festival in next {LOOKAHEAD_DAYS} days")
        
    return state


# ---------------------------------------------------------------------------
# 6.  Node 4 — check_holidays
#     Uses the `holidays` library to find the nearest public holiday.
# ---------------------------------------------------------------------------
def check_holidays_node(state: AgentState) -> AgentState:
    """
    Checks for US public holidays in the next 21 days.
    Writes: public_holiday, holiday_days_away.
    """
    today = date.fromisoformat(state["today_str"])
    public_holiday_name: str | None = None
    holiday_days_away: int = -1

    try:
        import holidays as hol_lib
        us_holidays = hol_lib.US(years=[today.year, today.year + 1])
        for delta in range(LOOKAHEAD_DAYS + 1):
            check_date = today + timedelta(days=delta)
            if check_date in us_holidays:
                public_holiday_name = us_holidays[check_date]
                holiday_days_away   = delta
                break
    except ImportError:
        public_holiday_name = "holidays library not installed"
        holiday_days_away   = -1

    state["public_holiday"]    = public_holiday_name
    state["holiday_days_away"] = holiday_days_away

    if public_holiday_name:
        print(f"[check_holidays] {public_holiday_name} in {holiday_days_away} day(s)")
    else:
        print(f"[check_holidays] No public holidays in next {LOOKAHEAD_DAYS} days")
    return state


# ---------------------------------------------------------------------------
# 7.  Node 5 — compute_lift
#     Applies the time-decay curve to calculate demand_lift_factor.
# ---------------------------------------------------------------------------
def compute_lift_node(state: AgentState) -> AgentState:
    """
    Time-decay formula:
        decay = max(0, 1 - (days_to_event / PROX_WINDOW_DAYS))
        lift  = 1.0 + (base_lift - 1.0) * decay

    Lift = 1.0 at day 7 (curve just begins).
    Lift = base_lift at day 0 (full peak on event day).
    No festival → lift stays at 1.0.
    """
    days_to   = state["days_to_event"]
    base_lift = state["base_lift"]

    if days_to >= 0:
        decay = max(0.0, 1.0 - (days_to / PROX_WINDOW_DAYS))
        lift  = round(1.0 + (base_lift - 1.0) * decay, 4)
    else:
        lift = 1.0

    state["demand_lift_factor"] = lift
    print(f"[compute_lift] days_to={days_to}  base_lift={base_lift}  → lift={lift}")
    return state


# ---------------------------------------------------------------------------
# 8.  Node 6 — assign_urgency
#     Deterministic urgency based on days_to_event.
#     Isolated so thresholds can be tuned independently of lift logic.
# ---------------------------------------------------------------------------
def assign_urgency_node(state: AgentState) -> AgentState:
    """
    Urgency rules (deterministic — NOT delegated to the LLM):
        0-1  days  → IMMEDIATE
        2-4  days  → HIGH
        5-10 days  → MEDIUM
        11+  days  → LOW
        no festival → LOW
    """
    days = state["days_to_event"]
    if days < 0:
        urgency = "LOW"
    elif days <= 1:
        urgency = "IMMEDIATE"
    elif days <= 4:
        urgency = "HIGH"
    elif days <= 10:
        urgency = "MEDIUM"
    else:
        urgency = "LOW"

    state["urgency"] = urgency
    print(f"[assign_urgency] {urgency}")
    return state


# ---------------------------------------------------------------------------
# 9.  Node 7 — call_llm
#     Passes all computed metrics to Gemini; receives proposal + justification.
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a festival & calendar pricing agent for a retail grocery store.

You receive pre-computed demand metrics for a single SKU.
Your job is to translate those metrics into a pricing proposal and clear justification.

Rules:
- If surcharge_exempt_triggered is true → suggested_action = "HOLD_EXEMPT" and explain that this category is protected from standard holiday surges.
- If demand_lift_factor > 1.15  → suggested_action = "SURCHARGE"
- If demand_lift_factor < 0.95  → suggested_action = "DISCOUNT"
- Otherwise                     → suggested_action = "HOLD"

- price_modifier = demand_lift_factor (use it directly as the multiplier)
- confidence_score:
    days_to_event 0-2   → 0.95
    days_to_event 3-5   → 0.80
    days_to_event 6-14  → 0.65
    no festival (-1)    → 0.40

Respond ONLY with a valid JSON object — no markdown fences, no preamble:
{
  "suggested_action": "<SURCHARGE | HOLD_EXEMPT | DISCOUNT | HOLD>",
  "price_modifier": <float>,
  "confidence_score": <float 0-1>,
  "headline": "<one-line summary>",
  "detailed_reasoning": "<multi-sentence explanation>"
}"""


def call_llm_node(state: AgentState) -> AgentState:
    """
    Sends computed metrics to Gemini. Receives suggested_action, price_modifier,
    confidence_score, headline, and detailed_reasoning.
    The urgency field is NOT sourced from the LLM — assign_urgency_node owns that.
    """
    row = state["current_row"]

    user_payload = json.dumps({
        "sku_id":               row["sku_id"],
        "product_name":         row["product_name"],
        "category":             row["category"],
        "today":                state["today_str"],
        "festival_name":        state["festival_name"],
        "days_to_event":        state["days_to_event"],
        "base_lift":            state["base_lift"],
        "demand_lift_factor":   state["demand_lift_factor"],
        "public_holiday":       state["public_holiday"],
        "holiday_days_away":    state["holiday_days_away"],
        "categories_affected":  state["categories_affected"],
        "surcharge_exempt_triggered": state["surcharge_exempt_triggered"]
    }, indent=2)

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=GEMINI_API_KEY,
        temperature=0.1,
    )

    # FIX 1: Use HumanMessage instead of a raw dictionary
    messages = [
        SystemMessage(content=SYSTEM_PROMPT), 
        HumanMessage(content=user_payload)
    ]
    
    raw_content = llm.invoke(messages).content

    # FIX 2: Safely extract the string if Gemini returns a list of blocks
    if isinstance(raw_content, list):
        raw_text = ""
        for block in raw_content:
            if isinstance(block, dict) and "text" in block:
                raw_text += block["text"]
            elif isinstance(block, str):
                raw_text += block
    else:
        raw_text = str(raw_content)
    # Safely extract string if Gemini returns a list of blocks

    # Strip markdown fences if present
    clean = re.sub(r"^```(?:json)?\s*", "", raw_text.strip(), flags=re.IGNORECASE)
    clean = re.sub(r"\s*```$", "", clean).strip()

    state["llm_response"] = json.loads(clean)
    print(f"[call_llm] action={state['llm_response']['suggested_action']}  modifier={state['llm_response']['price_modifier']}")
    return state

    

# ---------------------------------------------------------------------------
# 10. Node 8 — build_output
#     Assembles the final inter-agent JSON schema and appends to results.
# ---------------------------------------------------------------------------
def build_output_node(state: AgentState) -> AgentState:
    """
    Constructs the strict output schema.
    - urgency comes from assign_urgency_node (deterministic), NOT the LLM.
    - timestamp is injected here, never passed to the LLM.
    - status is always COMPLETED at this node.
    """
    row = state["current_row"]
    llm = state["llm_response"]

    output = {
        "agent_id":  "calendar",
        "sku_id":    row["sku_id"],
        "status":    "COMPLETED",
        "timestamp": "2026-05-26T12:00:00Z",  # ← injected here, not from LLM
        "metrics_evaluated": {
            "festival_name":       state["festival_name"],
            "days_to_event":       state["days_to_event"],
            "demand_lift_factor":  state["demand_lift_factor"],
            "public_holiday":      state["public_holiday"],
            "holiday_days_away":   state["holiday_days_away"],
            "categories_affected": state["categories_affected"],
            "surcharge_exempt_triggered": state["surcharge_exempt_triggered"],
        },
        "proposal": {
            "suggested_action": llm["suggested_action"],
            "price_modifier":   llm["price_modifier"],
            "confidence_score": llm["confidence_score"],
            "urgency":          state["urgency"],   # ← deterministic, not from LLM
        },
        "justification": {
            "headline":           llm["headline"],
            "detailed_reasoning": llm["detailed_reasoning"],
        },
    }

    state["results"].append(output)
    print(f"[build_output] [{row['sku_id']}] Proposal appended.")
    return state


# ---------------------------------------------------------------------------
# 11. Node 9 — advance_row
#     Increments the row cursor. Isolated so future enhancements
#     (rate-limiting, logging) can be added without touching business logic.
# ---------------------------------------------------------------------------
def advance_row_node(state: AgentState) -> AgentState:
    """Increments row_index to move to the next SKU."""
    state["row_index"] += 1
    return state


# ---------------------------------------------------------------------------
# 12. Conditional edges
# ---------------------------------------------------------------------------
def route_more_rows(state: AgentState) -> str:
    """Routes back to check_next_sku if rows remain, otherwise ends."""
    return "check_next_sku" if state["row_index"] < len(state["rows"]) else END


def route_has_festival(state: AgentState) -> str:
    """
    Gate: if no festival AND no holiday found, skip LLM and go straight
    to build_output with a HOLD proposal, preserving compute budget.
    """
    if state["festival_name"] is not None or (state["holiday_days_away"] >= 0):
        return "compute_lift"
    # No event found — still build a HOLD output but skip LLM
    return "compute_lift"   # always compute, LLM decides HOLD


# ---------------------------------------------------------------------------
# 13. Build the graph
# ---------------------------------------------------------------------------
def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("load_catalog",    load_catalog_node)
    graph.add_node("check_next_sku",  check_next_sku_node)
    graph.add_node("scan_festivals",  scan_festivals_node)
    graph.add_node("check_holidays",  check_holidays_node)
    graph.add_node("compute_lift",    compute_lift_node)
    graph.add_node("assign_urgency",  assign_urgency_node)
    graph.add_node("call_llm",        call_llm_node)
    graph.add_node("build_output",    build_output_node)
    graph.add_node("advance_row",     advance_row_node)

    graph.set_entry_point("load_catalog")
    graph.add_edge("load_catalog",   "check_next_sku")
    graph.add_edge("check_next_sku", "scan_festivals")
    graph.add_edge("scan_festivals", "check_holidays")
    graph.add_edge("check_holidays", "compute_lift")
    graph.add_edge("compute_lift",   "assign_urgency")
    graph.add_edge("assign_urgency", "call_llm")
    graph.add_edge("call_llm",       "build_output")
    graph.add_edge("build_output",   "advance_row")

    graph.add_conditional_edges("advance_row", route_more_rows, {
        "check_next_sku": "check_next_sku",
        END:              END,
    })

    return graph.compile()


# ---------------------------------------------------------------------------
# 14. Entry point
# ---------------------------------------------------------------------------
def main():
    import shutil
    import time # <-- Add this

    bootstrap_festival_json()

    # FIX 1: Always copy the freshest CSV, overwriting the old one
    SRC = Path("/mnt/user-data/uploads/calendar_data.csv")
    if SRC.exists():
        CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(SRC, CSV_PATH)
        print(f"[bootstrap] Refreshed CSV → {CSV_PATH}")

    app = build_graph()

    initial_state: AgentState = {
        "csv_path":            str(CSV_PATH),
        "today_str":           "2026-05-26",    # 1 day before Eid al-Adha
        "rows":                [],
        "row_index":           0,
        "current_row":         None,
        "festival_name":       None,
        "days_to_event":       -1,
        "base_lift":           1.0,
        "categories_affected": [],
        "public_holiday":      None,
        "holiday_days_away":   -1,
        "demand_lift_factor":  1.0,
        "urgency":             None,
        "llm_response":        None,
        "results":             [],
    }

    final_state = app.invoke(initial_state)

    print(f"\n✅ Done — {len(final_state['results'])} proposal(s) generated\n")
    for output in final_state["results"]:
        print(json.dumps(output, indent=2))
        print()

    # FIX 2: Generate a unique run ID so files never overwrite each other
    run_id = int(time.time()) 
    out_dir = BASE_DIR / "output"
    out_dir.mkdir(exist_ok=True)
    
    for output in final_state["results"]:
        # Appends the run_id to ensure a brand new file is created
        fname = out_dir / f"{output['sku_id']}_{initial_state['today_str']}_run{run_id}.json"
        with open(fname, "w") as fh:
            json.dump(output, fh, indent=2)
        print(f"[saved] → {fname}")


if __name__ == "__main__":
    main()
