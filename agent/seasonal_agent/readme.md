# Seasonal Agent

A small Python script that reads weather-driven input rows from `weather_input.csv`, evaluates each row with a simple seasonal pricing rule, and prints a JSON recommendation for each SKU.

## What it does

For each row in the CSV, `agent.py`:

- Reads `sku_id`, `current_temp_c`, `forecast_temp_c`, and `weather_event_flag`
- Maps the weather event to a price modifier
- Chooses a suggested action:
  - `SURCHARGE` for strong positive seasonal impact
  - `DISCOUNT` for negative impact
  - `HOLD` otherwise
- Prints a formatted JSON result to stdout

## Weather rules

- `Heatwave` -> price modifier `1.2` -> `SURCHARGE`
- `ColdSnap` -> price modifier `1.1` -> `HOLD`
- `Storm` -> price modifier `0.9` -> `DISCOUNT`
- Any other value -> price modifier `1.0` -> `HOLD`

## Files

- `agent.py` - main script
- `weather_input.csv` - sample input data

## Requirements

- Python 3.9 or newer

The script only uses standard library modules: `csv`, `json`, and `datetime`.

## Input format

`weather_input.csv` should use the following columns:

- `sku_id`
- `current_temp_c`
- `forecast_temp_c`
- `weather_event_flag`

Example:

```csv
sku_id,current_temp_c,forecast_temp_c,weather_event_flag
MEA001,32.8,35.0,Heatwave
MEA002,22.0,20.0,ColdSnap
MEA003,28.0,29.0,None
```

## Run it

From the project folder, run:

```bash
python agent.py
```

The script will process every row in `weather_input.csv` and print one JSON recommendation per record.

## Example output

```json
{
  "agent_id": "seasonal_intelligence",
  "sku_id": "MEA001",
  "status": "COMPLETED",
  "timestamp": "2026-06-15T12:00:00",
  "urgency": "MEDIUM",
  "metrics_evaluated": {
    "weather_event": "Heatwave",
    "current_temp": 32.8,
    "forecast_temp": 35.0
  },
  "proposal": {
    "suggested_action": "SURCHARGE",
    "price_modifier": 1.2,
    "confidence_score": 0.85
  },
  "justification": {
    "headline": "Weather impact detected",
    "detailed_reasoning": "Recommendation generated from seasonal weather analysis."
  }
}
```
