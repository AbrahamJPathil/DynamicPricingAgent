# Calendar & Festivity Agent

A LangGraph-based pricing agent that scans the next 21 days for festivals and
public holidays, applies a time-decay demand lift curve to each product SKU,
and outputs a structured JSON pricing proposal — ready to be consumed by a
larger multi-agent orchestrator.

---

## What It Does

For every SKU in your product catalog the agent:

1. Finds the nearest festival within 21 days that matches the SKU's category
2. Checks for US public holidays in the same window
3. Calculates how much demand lift to apply based on how close the event is
4. Decides whether the category is exempt from surcharging (e.g. staples during Eid)
5. Calls Gemini to write a pricing proposal and human-readable justification
6. Saves a timestamped JSON output file per SKU per run

---

## Project Structure

```
Calendar_agent/
├── calendar_agent.py          ← Main agent (run this)
├── requirements.txt           ← Python dependencies
├── .env                       ← Your Gemini API key (never commit this)
├── .env.example               ← Template for .env
├── data/
│   ├── calendar_data.csv      ← Product catalog (SKU, category, price, stock…)
│   └── festival_calendar.json ← Festival data with lift factors and categories
└── output/
    └── MEA003_2026-05-26_run1748123456.json   ← One file per SKU per run
```

---

## Prerequisites

- Python 3.10 or higher
- A Gemini API key — get one free at https://aistudio.google.com/app/apikey

---

## Setup

### 1. Create and activate a virtual environment

```powershell
# Windows
python -m venv .venv
.venv\Scripts\activate
```

```bash
# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

You should see `(.venv)` in your prompt.

### 2. Install dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Add your API key

```powershell
copy .env.example .env
```

Open `.env` and set your key:

```
GEMINI_API_KEY=AIzaSy...your-actual-key-here
```

Rules for the key value — no quotes, no spaces around `=`:
```
# ✅ Correct
GEMINI_API_KEY=AIzaSyABC123

# ❌ Wrong — quoted
GEMINI_API_KEY="AIzaSyABC123"

# ❌ Wrong — spaces
GEMINI_API_KEY = AIzaSyABC123
```

### 4. Add your data files

Place your product catalog CSV in `data/`:
```powershell
copy path\to\calendar_data.csv data\calendar_data.csv
```

Place the festival calendar JSON in `data/`:
```powershell
copy path\to\festival_calendar.json data\festival_calendar.json
```

The agent will exit with a clear error message if either file is missing.

---

## Running the Agent

```powershell
cd C:\Users\LENOVO\OneDrive\Desktop\Calendar_agent
.venv\Scripts\activate
python calendar_agent.py
```

### Expected terminal output

```
[bootstrap] Festival calendar already exists at ...\data\festival_calendar.json
[load_catalog] Loaded 3 SKU(s) from ...\data\calendar_data.csv
[check_next_sku] Processing SKU MEA003 — Premium Lamb Chops
[scan_festivals] Found: Eid al-Adha in 1 day(s) -> AFFECTED
[check_holidays] Juneteenth National Independence Day in 14 day(s)
[compute_lift] days_to=1  base_lift=1.75  → lift=1.6429
[assign_urgency] IMMEDIATE
[call_llm] action=SURCHARGE  modifier=1.6429
[build_output] [MEA003] Proposal appended.
[check_next_sku] Processing SKU SWT001 — Natural Medjool Dates
...
✅ Done — 3 proposal(s) generated
[saved] → output\MEA003_2026-05-26_run1748123456.json
```

---

## Output Schema

Each SKU produces one JSON file in `output/`. The filename includes the SKU,
the reference date, and a unique run timestamp so files never overwrite each other.

```json
{
  "agent_id": "calendar",
  "sku_id": "MEA003",
  "status": "COMPLETED",
  "timestamp": "2026-05-26T12:00:00Z",
  "metrics_evaluated": {
    "festival_name": "Eid al-Adha",
    "days_to_event": 1,
    "demand_lift_factor": 1.6429,
    "public_holiday": "Juneteenth National Independence Day",
    "holiday_days_away": 14,
    "categories_affected": ["meat", "sweets", "dairy", "bakery", "spices"],
    "surcharge_exempt_triggered": false
  },
  "proposal": {
    "suggested_action": "SURCHARGE",
    "price_modifier": 1.6429,
    "confidence_score": 0.95,
    "urgency": "IMMEDIATE"
  },
  "justification": {
    "headline": "Eid al-Adha is 1 day away — peak demand for meat products.",
    "detailed_reasoning": "With Eid al-Adha tomorrow, lamb and meat demand is at its seasonal peak..."
  }
}
```

### Field reference

| Field | Source | Description |
|---|---|---|
| `agent_id` | Hardcoded | Always `"calendar"` |
| `sku_id` | CSV | Product identifier |
| `status` | `build_output` node | Always `"COMPLETED"` on success |
| `timestamp` | `build_output` node | UTC time of output assembly |
| `festival_name` | `scan_festivals` node | Nearest matching festival, or `null` |
| `days_to_event` | `scan_festivals` node | Days until festival; `-1` if none found |
| `demand_lift_factor` | `compute_lift` node | Calculated multiplier from decay curve |
| `surcharge_exempt_triggered` | `scan_festivals` node | `true` if this category is exempt |
| `suggested_action` | LLM (`call_llm` node) | `SURCHARGE`, `HOLD`, `DISCOUNT`, or `HOLD_EXEMPT` |
| `price_modifier` | LLM | Equal to `demand_lift_factor` |
| `confidence_score` | LLM | 0.40–0.95 based on days to event |
| `urgency` | `assign_urgency` node | Deterministic — never from the LLM |
| `headline` | LLM | One-line summary |
| `detailed_reasoning` | LLM | Multi-sentence explanation |

---

## How the Demand Lift Works

The lift starts building 7 days before an event and peaks on the event day:

```
decay = max(0, 1 - (days_to_event / 7))
lift  = 1.0 + (base_lift - 1.0) × decay
```

Example for Eid al-Adha (`base_lift = 1.75`):

| Days to Event | Lift Factor | Action |
|---|---|---|
| 7 | 1.000 | HOLD |
| 5 | 1.214 | SURCHARGE |
| 3 | 1.429 | SURCHARGE |
| 1 | 1.643 | SURCHARGE |
| 0 | 1.750 | SURCHARGE (peak) |

---

## Surcharge Exemption Logic

Some categories see demand lift during a festival but should not be surcharged
— typically everyday staples (produce, spices) where a price hike would unfairly
burden low-income households during a religious holiday.

In `festival_calendar.json` this is controlled by two fields:

```json
{
  "name": "Eid al-Adha",
  "categories": ["meat", "sweets", "dairy", "bakery", "spices", "beverages", "snacks"],
  "surcharge_exempt": ["produce", "snacks"]
}
```

**Rule:** Every category in `surcharge_exempt` MUST also be in `categories`.
`surcharge_exempt` controls the *pricing action*, not the *visibility* of the festival.
If a category is only in `surcharge_exempt` and not in `categories`, the agent
will not find the festival at all for that SKU.

When `surcharge_exempt_triggered = true`:
- `base_lift` is forced to `1.0` in `scan_festivals`
- `demand_lift_factor` stays at `1.0` through `compute_lift`
- LLM outputs `suggested_action = "HOLD_EXEMPT"` with an explanation

---

## Graph Architecture

The agent is a LangGraph `StateGraph` with 9 nodes. Each business concern is
isolated in its own node so individual rules can be tuned without touching the rest.

```
load_catalog
     │
check_next_sku ──(no more rows)──► END
     │
scan_festivals        ← reads festival_calendar.json, matches by category
     │
check_holidays        ← reads US public holidays via `holidays` library
     │
compute_lift          ← applies time-decay formula, produces demand_lift_factor
     │
assign_urgency        ← deterministic if/elif — never delegated to LLM
     │
call_llm              ← sends pre-computed metrics to Gemini, gets proposal
     │
build_output          ← assembles final JSON schema, injects timestamp
     │
advance_row           ← increments cursor
     │
(loop back to check_next_sku)
```

### Why urgency is deterministic

The orchestrator uses `urgency` to route `IMMEDIATE` proposals to human review
queues. If the LLM owned this field, the same inputs could produce different
urgency levels on different runs, breaking routing consistency.

### Why the LLM receives pre-computed numbers

All metrics (`demand_lift_factor`, `days_to_event`, `surcharge_exempt_triggered`)
are calculated before the LLM is called. The LLM's job is reasoning and
language — not arithmetic. This prevents hallucination of core metrics while
still using the model for human-readable justification.

---

## Changing the Reference Date

The `today_str` in `main()` controls what date the agent treats as "today".
Change it to test any scenario:

```python
initial_state: AgentState = {
    "today_str": "2026-11-19",   # ← 7 days before Thanksgiving
    ...
}
```

---

## Festival Calendar JSON Format

```json
{
  "name": "Eid al-Adha",
  "month": 5,
  "day": 27,
  "date_hint": "2026-05-27",
  "base_lift": 1.75,
  "categories": ["meat", "sweets", "dairy", "bakery", "spices", "beverages", "snacks"],
  "surcharge_exempt": ["produce", "snacks"],
  "region": "global"
}
```

| Field | Description |
|---|---|
| `date_hint` | Exact Gregorian date for floating holidays (Eid, Diwali). Year is swapped at runtime. Takes priority over `month`/`day`. |
| `month` + `day` | Used for fixed annual holidays (Christmas = 12/25) when `date_hint` is `null`. |
| `base_lift` | Peak demand multiplier on the event day. |
| `categories` | All categories that see ANY demand change from this festival. |
| `surcharge_exempt` | Subset of `categories` that should receive `HOLD_EXEMPT` instead of `SURCHARGE`. |
| `region` | Informational in v2. Future versions can use for regional deployments. |

---

## Product Catalog CSV Format

The agent reads these columns from `calendar_data.csv`:

| Column | Used by | Notes |
|---|---|---|
| `sku_id` | All nodes | Primary identifier |
| `product_name` | `check_next_sku`, `call_llm` | Sent to LLM for context |
| `category` | `scan_festivals` | Must match a value in the festival's `categories` list |

All other columns in the CSV (price, stock, expiry, etc.) are ignored by this
agent but preserved for the orchestrator.

---

## Integrating with the Orchestrator

Call `run_agent_for_sku()` directly, or invoke the compiled graph:

```python
from calendar_agent import build_graph, AgentState

app = build_graph()

result = app.invoke({
    "csv_path":            "data/calendar_data.csv",
    "today_str":           "2026-11-19",
    "rows":                [],
    "row_index":           0,
    "current_row":         None,
    "festival_name":       None,
    "days_to_event":       -1,
    "base_lift":           1.0,
    "categories_affected": [],
    "surcharge_exempt_triggered": False,
    "public_holiday":      None,
    "holiday_days_away":   -1,
    "demand_lift_factor":  1.0,
    "urgency":             None,
    "llm_response":        None,
    "results":             [],
})

proposals = result["results"]   # list of dicts, one per SKU
```

The orchestrator is responsible for:
- Enforcing a `price_modifier` floor (never below `cost_price`)
- Weighing this agent's signal against Inventory, Competitor, and Social agents
- Routing `IMMEDIATE` + `price_modifier > 1.5` proposals to human review

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `GEMINI_API_KEY not set` | `.env` missing or key not set | Create `.env` from `.env.example` and add your key |
| `festival_calendar.json is missing` | JSON not in `data/` | Copy the file into `data/` |
| `FileNotFoundError: calendar_data.csv` | CSV not in `data/` | Copy CSV into `data/` |
| `401 UNAUTHENTICATED` | Wrong or invalid API key | Get a new key from https://aistudio.google.com/app/apikey |
| `429 RESOURCE_EXHAUSTED` | Free tier quota used up | Wait until midnight Pacific, or switch model to `gemini-1.5-flash` |
| `ModuleNotFoundError` | venv not active or deps not installed | Activate `.venv` then `pip install -r requirements.txt` |
| `NoneType is not subscriptable` in `build_output` | `call_llm` returned without saving to state | Ensure `state["llm_response"] = json.loads(clean)` is the last line before `return state` in `call_llm_node` |
| Festival not found for a SKU | Category missing from `categories` in JSON | Add the category to the festival's `categories` array in `festival_calendar.json` |

### Switching models to avoid quota errors

In `call_llm_node`, change the model name:

```python
llm = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",   # ← change here
    ...
)
```

Available models: `gemini-2.5-flash`, `gemini-2.0-flash`, `gemini-1.5-flash`, `gemini-1.5-pro`
