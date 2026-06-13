import csv
from orchestrator.graph import pricing_graph

def run_all():
    with open("data/inputs/products.csv") as f:
        for row in csv.DictReader(f):
            initial_state = {
                "product_id":        row["product_id"],
                "inventory_output":  None,
                "weather_output":    None,
                "social_output":     None,
                "competitor_output": None,
                "final_price":       None,
                "reasoning":         None,
            }
            pricing_graph.invoke(initial_state)

if __name__ == "__main__":
    run_all()