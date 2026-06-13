from langgraph.graph import StateGraph, END
from shared.schemas.state import PricingState

from agents.inventory.agent import inventory_node
from agents.weather.agent import weather_node
from agents.social.agent import social_node
from agents.competitor.agent import competitor_node


def orchestrator_node(state: PricingState) -> dict:
    """
    Placeholder — receives all sub-agent outputs.
    Add pricing logic here later.
    """
    print("Orchestrator received:")
    for key in ["inventory_output", "weather_output", "social_output", "competitor_output"]:
        output = state.get(key)
        if output:
            print(f"  [{output['agent_id']}] {output['data']}")
    return {}


def build_graph():
    graph = StateGraph(PricingState)

    graph.add_node("inventory",   inventory_node)
    graph.add_node("weather",     weather_node)
    graph.add_node("social",      social_node)
    graph.add_node("competitor",  competitor_node)
    graph.add_node("orchestrator", orchestrator_node)

    graph.set_entry_point("inventory")

    graph.add_edge("inventory",   "orchestrator")
    graph.add_edge("weather",     "orchestrator")
    graph.add_edge("social",      "orchestrator")
    graph.add_edge("competitor",  "orchestrator")
    graph.add_edge("orchestrator", END)

    return graph.compile()


pricing_graph = build_graph()