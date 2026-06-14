from datetime import datetime

def seasonal_agent(state):

    weather_event = state["weather_event"]

    if weather_event == "Heatwave":
        multiplier = 1.2

    elif weather_event == "ColdSnap":
        multiplier = 1.1

    elif weather_event == "Storm":
        multiplier = 0.9

    else:
        multiplier = 1.0

    if multiplier > 1.1:
        action = "SURCHARGE"

    elif multiplier < 1.0:
        action = "DISCOUNT"

    else:
        action = "HOLD"

    return {
        "agent_id": "seasonal_intelligence",
        "sku_id": state["sku_id"],
        "status": "COMPLETED",
        "timestamp": datetime.now().isoformat(),
        "urgency": "MEDIUM",
        "metrics_evaluated": {
            "weather_event": weather_event,
            "current_temp": state["current_temp"],
            "forecast_temp": state["forecast_temp"]
        },
        "proposal": {
            "suggested_action": action,
            "price_modifier": multiplier,
            "confidence_score": 0.85
        },
        "justification": {
            "headline": "Weather impact detected",
            "detailed_reasoning":
            "Recommendation generated from seasonal weather analysis."
        }
    }
  
