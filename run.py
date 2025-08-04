from src.analysis.deep_change_point import NeuralChangePointModel
from src.analysis.causal_analysis import CausalImpactAnalyzer
import pandas as pd

def main():
    # Load data
    prices = pd.read_csv("data/raw/brent_prices.csv")
    events = pd.read_csv("data/external/geopolitical/key_events.csv")
    
    # Run analysis
    cp_model = NeuralChangePointModel(input_shape=(30, 1))
    cp_results = cp_model.detect_changepoints(prices)
    
    causal_analyzer = CausalImpactAnalyzer(prices, events)
    effects = causal_analyzer.estimate_effect("OPEC_Decisions")
    
    print(f"Detected change points: {cp_results['changepoints']}")
    print(f"OPEC decision impact: /bbl")

if __name__ == "__main__":
    main()
