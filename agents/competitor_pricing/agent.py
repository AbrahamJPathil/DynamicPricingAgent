import os, json, math, requests, pandas as pd
import logging
from dotenv import load_dotenv
from datetime import datetime, timezone
from typing import TypedDict, Optional
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from langgraph.graph import StateGraph
try:
    from IPython.display import Image, display
    _IPYTHON = True
except ImportError:
    _IPYTHON = False

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
# ─────────────────────────────────────────────
#  CONFIGURATION  (set your creds here or via env vars)
# ─────────────────────────────────────────────
KROGER_CLIENT_ID = os.getenv("KROGER_CLIENT_ID")
KROGER_CLIENT_SECRET = os.getenv("KROGER_CLIENT_SECRET")
KROGER_LOCATION_ID   = os.getenv("KROGER_LOCATION_ID",   "01400943")   # default: a Kroger store

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
CSV_PATH = os.path.join(ROOT, "data", "inputs", "competitor", "data.csv")
OUTPUT_PATH = os.path.join(ROOT, "data", "outputs", "pricing_report.json")

KROGER_TOKEN_URL   = "https://api.kroger.com/v1/connect/oauth2/token"
KROGER_PRODUCT_URL = "https://api.kroger.com/v1/products"


# ─────────────────────────────────────────────
#  AGENT STATE
# ─────────────────────────────────────────────
class AgentState(TypedDict):
    # Inputs
    csv_path:      str
    location_id:   str

    # Intermediate
    products:      list          # rows from CSV as dicts
    access_token:  Optional[str]
    auth_error:    Optional[str]
    raw_prices:    dict          # sku_id → kroger price (float | None)

    # Outputs
    results:       list          # final list of report dicts
    report_path:   Optional[str]
    errors:        list          # non-fatal per-SKU errors


# ─────────────────────────────────────────────
#  NODE 1 — LOAD CSV
# ─────────────────────────────────────────────
def load_csv(state: AgentState) -> AgentState:
    """Read products.csv → list of product dicts."""
    logger.info("[NODE 1/5]   Loading product catalogue from CSV")
    try:
        df = pd.read_csv(state["csv_path"])
        required = {"sku_id", "product_name", "unit", "our_price"}
        missing  = required - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing columns: {missing}")

        products = df.to_dict(orient="records")
        logger.info(
            f"Loaded {len(products)} SKU(s): {[p['sku_id'] for p in products]}"
        )
        return {**state, "products": products, "errors": []}

    except Exception as exc:
        logger.error(f"CSV error: {exc}")
        return {**state, "products": [], "errors": [str(exc)]}


# ─────────────────────────────────────────────
#  NODE 2 — AUTH KROGER(GET TOKEN)
# ─────────────────────────────────────────────
def auth_kroger(state: AgentState) -> AgentState:
    """Obtain a client-credentials bearer token from Kroger."""
    logger.info("[NODE 2/5]   Authenticating with Kroger API …")

    if not KROGER_CLIENT_ID or not KROGER_CLIENT_SECRET:
        msg = "Missing Kroger API credentials in .env file."
        logger.error(msg)
        return {**state, "access_token": None, "auth_error": msg}

    try:
        resp = requests.post(
            KROGER_TOKEN_URL,
            data={
                "grant_type":    "client_credentials",
                "scope":         "product.compact",
            },
            auth=(KROGER_CLIENT_ID, KROGER_CLIENT_SECRET),
            timeout=10,
            verify=False
        )
        resp.raise_for_status()
        token = resp.json()["access_token"]
        logger.info("Token obtained successfully.")
        return {**state, "access_token": token, "auth_error": None}

    except Exception as exc:
        msg = f"Kroger auth failed: {exc}"
        logger.error(msg)
        return {**state, "access_token": None, "auth_error": msg}


# ─────────────────────────────────────────────
#  NODE 3 — FETCH COMPETITOR PRICES
# ─────────────────────────────────────────────

def _fetch_kroger_price(product: dict, token: str, location_id: str) -> Optional[float]:
    """Search Kroger's catalogue by product name and return the first shelf price."""
    headers = {"Authorization": f"Bearer {token}"}
    search_term = f"{product['product_name']} {product['unit']}"

    params = {
        "filter.term": search_term,
        "filter.locationId": location_id,
        "filter.limit": 1,
    }
    try:
        resp = requests.get(KROGER_PRODUCT_URL, headers=headers,
                            params=params, timeout=10, verify=False)
        resp.raise_for_status()
        items = resp.json().get("data", [])
        items = resp.json().get("data", [])

        logger.info(f"Search Term: {search_term}")

        for i, item in enumerate(items):
            logger.info(
                f"Result {i+1}: {item.get('description')}"
            )

        if not items:
            return None
        prices = items[0].get("items", [{}])[0].get("price", {})
        # prefer "promo" price if available, otherwise "regular"
        return float(prices.get("promo") or prices.get("regular") or 0) or None
    except Exception:
        return None


def fetch_prices(state: AgentState) -> AgentState:
    """For each product, query Kroger and store the price."""
    logger.info("[NODE 3/5]   Fetching Kroger competitor prices …")

    token      = state.get("access_token")
    raw_prices = {}
    errors     = list(state.get("errors", []))

    for product in state["products"]:
        sku  = product["sku_id"]
        
        if not token:
            price = None
            tag = "NO-TOKEN"
        else:
            price = _fetch_kroger_price(
                product,
                token,
                state["location_id"]
            )
            tag = "LIVE"

        raw_prices[sku] = price
        status = f"${price:.2f}" if price else "N/A"
        logger.info(
            f"{sku} -> Kroger Price: {status} [{tag}]"
        )

        if price is None:
            errors.append(f"{sku}: Could not retrieve Kroger price.")

    return {**state, "raw_prices": raw_prices, "errors": errors}


# ─────────────────────────────────────────────
#  NODE 4 — ANALYSE & BUILD PROPOSALS
# ─────────────────────────────────────────────
def _build_proposal(our_price: float, comp_price: float):
    """
    Compare prices and return (action, modifier, confidence, reasoning).

    modifier  = comp_price / our_price  → multiply our_price by modifier
                to land exactly at competitor price.
    """
    diff_pct = abs(our_price - comp_price) / comp_price * 100

    if math.isclose(our_price, comp_price, rel_tol=1e-3):
        action     = "HOLD"
        modifier   = 1.0
        confidence = 0.95
        reasoning  = (
            f"Our price (${our_price:.2f}) matches Kroger's (${comp_price:.2f}). "
            "No price change needed — maintain current positioning."
        )
    elif our_price > comp_price:
        action     = "DISCOUNT"
        modifier   = round(comp_price / our_price, 4)
        confidence = round(min(0.99, 0.70 + diff_pct / 100), 2)
        reasoning  = (
            f"Our price (${our_price:.2f}) exceeds Kroger's (${comp_price:.2f}) "
            f"by {diff_pct:.1f}%. Applying a {(1-modifier)*100:.1f}% discount "
            "brings us level with the competitor and protects market share."
        )
    else:  # our_price < comp_price
        action     = "SURCHARGE"
        modifier   = round(comp_price / our_price, 4)
        confidence = round(min(0.99, 0.60 + diff_pct / 100), 2)
        reasoning  = (
            f"Kroger prices (${comp_price:.2f}) are higher than ours (${our_price:.2f}) "
            f"by {diff_pct:.1f}%. A {(modifier-1)*100:.1f}% surcharge captures "
            "margin while remaining competitive."
        )

    return action, modifier, confidence, reasoning


HEADLINE_MAP = {
    "HOLD":      "Price Parity Achieved",
    "DISCOUNT":  "Competitive Pressure Detected",
    "SURCHARGE": "Margin Expansion Opportunity",
}

def analyze(state: AgentState) -> AgentState:
    """Compare our prices vs Kroger and generate proposals for each SKU."""
    logger.info("[NODE 4/5]   Analysing prices and generating proposals …")

    results    = []
    errors     = list(state.get("errors", []))
    timestamp  = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    for product in state["products"]:
        sku       = product["sku_id"]
        our_price = float(product["our_price"])
        comp_price = state["raw_prices"].get(sku)

        if comp_price is None:
            status = "FAILED"
            entry  = {
                "agent_id":         "competitor_pricing",
                "status":           status,
                "timestamp":        timestamp,
                "metrics_evaluated": {
                    "sku":               sku,
                    "our_current_price": our_price,
                    "competitor_price":  None,
                },
                "proposal": None,
                "justification": {
                    "headline":          "Data Unavailable",
                    "detailed_reasoning": f"Could not retrieve Kroger price for SKU {sku}.",
                },
            }
        else:
            action, modifier, confidence, reasoning = _build_proposal(our_price, comp_price)
            entry = {
                "agent_id":  "competitor_pricing",
                "status":    "COMPLETED",
                "timestamp": timestamp,
                "metrics_evaluated": {
                    "sku":               sku,
                    "our_current_price": round(our_price, 2),
                    "competitor_price":  round(comp_price, 2),
                },
                "proposal": {
                    "suggested_action": action,
                    "price_modifier":   modifier,
                    "confidence_score": confidence,
                },
                "justification": {
                    "headline":          HEADLINE_MAP[action],
                    "detailed_reasoning": reasoning,
                },
            }
            logger.info(
                f"{sku} -> {action} "
                f"(modifier={modifier}, confidence={confidence})"
            )

        results.append(entry)

    return {**state, "results": results, "errors": errors}


# ─────────────────────────────────────────────
#  NODE 5 — COMPILE REPORT
# ─────────────────────────────────────────────
def compile_report(state: AgentState) -> AgentState:
    """Write JSON report to disk and print a summary table."""
    logger.info("[NODE 5/5]   Compiling final report …")

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        json.dump(state["results"], f, indent=2)

    # ── Pretty summary table ─────────────────────────────────────────────
    print()
    print("┌" + "─"*76 + "┐")
    print(f"│{'COMPETITOR PRICING REPORT':^76}│")
    print("├" + "─"*20 + "┬" + "─"*10 + "┬" + "─"*10 + "┬" + "─"*12 + "┬" + "─"*20 + "┤")
    print(f"│{'SKU':<20}│{'Our $':>10}│{'Kroger $':>10}│{'Action':>12}│{'Modifier':>20}│")
    print("├" + "─"*20 + "┼" + "─"*10 + "┼" + "─"*10 + "┼" + "─"*12 + "┼" + "─"*20 + "┤")

    for r in state["results"]:
        m    = r["metrics_evaluated"]
        prop = r.get("proposal") or {}
        sku  = m["sku"]
        our  = f"${m['our_current_price']:.2f}" if m["our_current_price"] else "N/A"
        comp = f"${m['competitor_price']:.2f}"  if m["competitor_price"]  else "N/A"
        act  = prop.get("suggested_action", "N/A")
        mod  = str(prop.get("price_modifier", "N/A"))
        print(f"│{sku:<20}│{our:>10}│{comp:>10}│{act:>12}│{mod:>20}│")

    print("└" + "─"*20 + "┴" + "─"*10 + "┴" + "─"*10 + "┴" + "─"*12 + "┴" + "─"*20 + "┘")

    if state.get("errors"):
        logger.warning("Non-fatal errors encountered:")
        for e in state["errors"]:
            logger.warning(e)

    logger.info(f"Report saved to {OUTPUT_PATH}")
    return {**state, "report_path": OUTPUT_PATH}


# ─────────────────────────────────────────────
#  BUILD THE LANGGRAPH
# ─────────────────────────────────────────────
graph = StateGraph(AgentState)

graph.add_node("load_csv",       load_csv)
graph.add_node("auth_kroger",    auth_kroger)
graph.add_node("fetch_prices",   fetch_prices)
graph.add_node("analyze",        analyze)
graph.add_node("compile_report", compile_report)

graph.set_entry_point("load_csv")
graph.add_edge("load_csv",       "auth_kroger")
graph.add_edge("auth_kroger",    "fetch_prices")
graph.add_edge("fetch_prices",   "analyze")
graph.add_edge("analyze",        "compile_report")
graph.set_finish_point("compile_report")

app = graph.compile()


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ── Visualise the graph (works in Jupyter; prints ASCII fallback in CLI) ─
    try:
        if _IPYTHON:
            display(Image(app.get_graph().draw_mermaid_png()))
        else:
            logger.info("Graph structure:")
            logger.info(app.get_graph().draw_mermaid())
    except Exception as e:
        logger.warning(f"Graph render skipped: {e}")

    logger.info("COMPETITOR PRICING AGENT - Starting run")

    initial_state: AgentState = {
        "csv_path":     CSV_PATH,
        "location_id":  KROGER_LOCATION_ID,
        "products":     [],
        "access_token": None,
        "auth_error":   None,
        "raw_prices":   {},
        "results":      [],
        "report_path":  None,
        "errors":       [],
    }

    final_state = app.invoke(initial_state)

    logger.info("FINAL JSON OUTPUT")
    logger.info(json.dumps(final_state["results"], indent=2))