"""
Inventory agent — template for all sub-agents.

YOUR JOB AS A SUB-AGENT DEVELOPER:
  1. Load your CSV from data/inputs/inventory/
  2. Extract whatever fields are relevant
  3. Put them in the `data` dict and return

Do NOT touch anything outside this folder.
Do NOT write pricing logic here — that's the orchestrator's job later.
"""

import csv
from datetime import datetime, timezone
from pathlib import Path

from shared.schemas.state import PricingState, AgentOutput

INPUT_DIR = Path("data/inputs/inventory")


def _load_csv(product_id: str) -> dict:
    path = INPUT_DIR / f"{product_id}.csv"
    if not path.exists():
        return {}
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    return rows[-1] if rows else {}


def inventory_node(state: PricingState) -> dict:
    row = _load_csv(state["product_id"])

    output: AgentOutput = {
        "agent_id":  "inventory",
        "data":      dict(row),       # pass the raw CSV row up as-is
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    return {"inventory_output": output}