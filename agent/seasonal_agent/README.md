# Seasonal Intelligence Agent

This project contains a LangGraph-based agent in [agent_llm.py](agent_llm.py) that reads a single input row, derives weather and demand signals, calls Gemini, and returns a structured pricing recommendation.

## Workflow

The runtime flow is:

```mermaid
flowchart LR
  A[seasonal_agent(row)] --> B[analyze_weather]
  B --> C[compute_demand]
  C --> D[compute_velocity]
  D --> E[assign_urgency]
  E --> F[call_llm]
  F --> G[build_output]
  G --> H[final_output]
```

`seasonal_agent(row)` is the entry point other code can call. It sends the row into the compiled LangGraph, and the graph returns the `final_output` object.

## Node-by-node explanation

### `analyze_weather`

This is the first node in the graph.

What it does:

- Reads `current_row` from the incoming state
- Copies `sku_id` into the working state
- Converts `current_temp_c` and `forecast_temp_c` to floats
- Copies `weather_event_flag` into `weather_event`

Why it exists:

- It normalizes the raw row into typed fields the downstream nodes can use
- It separates raw CSV-style input from the graph’s internal state

### `compute_demand`

This node calculates `demand_multiplier`.

What it does:

- Reads `weather_sensitivity_score` from the input row
- Starts with a baseline multiplier of `1.0`
- Adjusts the multiplier based on weather:
  - `Heatwave` increases demand by `sensitivity * 0.3`
  - `ColdSnap` increases demand by `sensitivity * 0.1`
  - `Storm` decreases demand by `sensitivity * 0.2`
- Rounds the result to two decimals

Why it exists:

- It converts weather conditions into a demand impact signal
- That signal is later used to determine urgency and shape the prompt sent to Gemini

### `compute_velocity`

This node calculates sales momentum.

What it does:

- Reads `units_sold_last_24h`
- Reads `avg_units_sold_same_week_last_year`
- Computes a ratio as `current_sales / max(baseline, 1)`
- Rounds the result to two decimals

Why it exists:

- It gives the agent a simple sales velocity signal
- The `max(..., 1)` guard prevents division by zero when the historical baseline is missing or zero

### `assign_urgency`

This node converts the demand signal into a coarse urgency level.

What it does:

- Uses `demand_multiplier`
- Sets urgency to:
  - `HIGH` when the score is at least `1.3`
  - `MEDIUM` when the score is at least `1.1`
  - `LOW` otherwise

Why it exists:

- It turns a numeric score into a business-friendly label
- That label is later passed into the LLM prompt and included in the final output

### `call_llm`

This node sends the structured context to Gemini.

What it does:

- Builds a prompt containing:
  - `weather_event`
  - `current_temp`
  - `forecast_temp`
  - `sales_velocity_ratio`
  - `demand_multiplier`
  - `urgency`
- Calls `client.models.generate_content` with `gemini-2.5-flash`
- Strips code fences if Gemini returns JSON inside triple backticks
- Parses the response text with `json.loads`
- Stores the parsed payload in `llm_output`

Why it exists:

- This is the reasoning node that turns the calculated signals into an LLM-generated recommendation
- The code expects Gemini to return JSON with these keys:
  - `suggested_action`
  - `price_modifier`
  - `confidence_score`
  - `headline`
  - `detailed_reasoning`

### `build_output`

This node assembles the final response object.

What it does:

- Reads `llm_output`
- Builds a final JSON-ready dictionary with:
  - `agent_id`
  - `sku_id`
  - `status`
  - `timestamp`
  - `metrics_evaluated`
  - `proposal`
  - `justification`

Why it exists:

- It separates the model response from the application response contract
- It produces the stable output shape the caller receives

## Code structure

### Imports

The script imports:

- `os`, `json`, and `re` for configuration and parsing
- `datetime` for timestamps
- `TypedDict` for the LangGraph state definition
- `load_dotenv` for `.env` loading
- `StateGraph` from LangGraph to define the workflow
- `genai.Client` and `ChatGoogleGenerativeAI` from Google Gemini packages

### Environment loading

The script calls `load_dotenv()` so `GEMINI_API_KEY` can be loaded from a local `.env` file.

The current code also prints debug messages before and after loading the environment and when the Gemini client is created. Those prints are useful during development, but they make the script noisy in production.

### State definition

`SeasonalState` defines the data the graph passes between nodes. It includes:

- Raw input: `current_row`
- Normalized weather fields: `sku_id`, `current_temp`, `forecast_temp`, `weather_event`
- Derived metrics: `sales_velocity_ratio`, `demand_multiplier`, `urgency`
- LLM response: `llm_output`
- Final response: `final_output`

## Graph wiring

The graph is built in this order:

1. `analyze_weather`
2. `compute_demand`
3. `compute_velocity`
4. `assign_urgency`
5. `call_llm`
6. `build_output`

That chain is configured with `add_node`, `set_entry_point`, `add_edge`, and `set_finish_point`, then compiled into `seasonal_graph`.

## Entry point

`seasonal_agent(row)` is the function the orchestrator should call.

It:

- Wraps the incoming row in the graph state under `current_row`
- Invokes the compiled graph
- Returns only the `final_output` dictionary

## Expected input shape

The graph expects each row to contain at least these fields:

- `sku_id`
- `current_temp_c`
- `forecast_temp_c`
- `weather_event_flag`
- `weather_sensitivity_score`
- `units_sold_last_24h`
- `avg_units_sold_same_week_last_year`

Example row:

```python
{
    "sku_id": "MEA001",
    "current_temp_c": 32.8,
    "forecast_temp_c": 35.0,
    "weather_event_flag": "Heatwave",
    "weather_sensitivity_score": 0.9,
    "units_sold_last_24h": 150,
    "avg_units_sold_same_week_last_year": 100
}
```

## Example output shape

The returned object looks like this:

```json
{
  "agent_id": "seasonal_intelligence",
  "sku_id": "MEA001",
  "status": "COMPLETED",
  "timestamp": "2026-06-15T12:00:00",
  "metrics_evaluated": {
    "urgency": "MEDIUM",
    "weather_event": "Heatwave",
    "current_temp": 32.8,
    "forecast_temp": 35.0,
    "sales_velocity_ratio": 1.5
  },
  "proposal": {
    "suggested_action": "SOME_ACTION",
    "price_modifier": 1.2,
    "confidence_score": 0.85
  },
  "justification": {
    "headline": "Weather impact detected",
    "detailed_reasoning": "..."
  }
}
```

## Requirements

- Python 3.9 or newer
- `langgraph`
- `langchain-google-genai`
- `google-genai`
- `python-dotenv`
- A valid `GEMINI_API_KEY` in `.env` or the environment

## Run it

The `seasonal_agent` function is designed to be imported and called by another script or orchestrator.

If you want to test the graph directly, you can use the commented example near the bottom of [agent_llm.py](agent_llm.py).

## Notes

- `agent_llm.py` currently contains several debug `print` statements.
- The prompt string in `call_llm` includes a stray `print("Sending Gemini request")` line inside the triple-quoted prompt text. That text is sent to the model as part of the prompt, not executed by Python.
- The `ChatGoogleGenerativeAI` import is present but not used in the current implementation.
