from typing import TypedDict, Optional


class AgentOutput(TypedDict):
    agent_id: str
    data: dict          # raw signals — each agent decides what goes here
    timestamp: str


class PricingState(TypedDict):
    product_id: str

    # Sub-agents write into their own key; None until that agent runs
    inventory_output:  Optional[AgentOutput]
    weather_output:    Optional[AgentOutput]
    social_output:     Optional[AgentOutput]
    competitor_output: Optional[AgentOutput]

    # Orchestrator will use these later — left empty for now
    final_price:  Optional[float]
    reasoning:    Optional[str]