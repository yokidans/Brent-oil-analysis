# src/dashboard/backend/app.py
from fastapi import FastAPI
from strawberry.fastapi import GraphQLRouter
import strawberry
from typing import List, Optional
import pandas as pd
from src.analysis.deep_change_point import NeuralChangePointModel
from src.analysis.causal_analysis import CausalImpactAnalyzer

app = FastAPI(title="Advanced Oil Analytics API")

# GraphQL Type Definitions
@strawberry.type
class ChangePoint:
    date: str
    probability: float
    mean_before: float
    mean_after: float
    events: List[str]

@strawberry.type
class CausalEffect:
    treatment: str
    estimate: float
    ci_lower: float
    ci_upper: float

@strawberry.type
class Query:
    @strawberry.field
    def changepoints(self, lookback: Optional[int] = 365) -> List[ChangePoint]:
        model = NeuralChangePointModel(input_shape=(30, 1))
        results = model.detect_changepoints(price_data)
        return [
            ChangePoint(
                date=cp['date'],
                probability=cp['probability'],
                mean_before=cp['mean_before'],
                mean_after=cp['mean_after'],
                events=cp['events']
            )
            for cp in results['changepoints']
        ]
    
    @strawberry.field
    def causal_effects(self, treatment: str) -> CausalEffect:
        analyzer = CausalImpactAnalyzer(price_data, event_data)
        result = analyzer.estimate_effect(treatment)
        return CausalEffect(
            treatment=treatment,
            estimate=result.estimate,
            ci_lower=result.conf_int[0],
            ci_upper=result.conf_int[1]
        )

# GraphQL Schema
schema = strawberry.Schema(Query)
graphql_app = GraphQLRouter(schema)

app.include_router(graphql_app, prefix="/graphql")

# REST Endpoints
@app.get("/neural_cp/visualization")
async def get_neural_visualization():
    """Returns visualization data for neural change points"""
    model = NeuralChangePointModel(input_shape=(30, 1))
    results = model.detect_changepoints(price_data)
    
    # Process attention weights for visualization
    attention = model.get_attention_weights(price_data)
    
    return {
        "changepoints": results['changepoints'],
        "attention": attention,
        "latent_space": model.get_latent_representation(price_data)
    }

@app.get("/causal_network")
async def get_causal_network():
    """Returns causal graph structure"""
    analyzer = CausalImpactAnalyzer(price_data, event_data)
    return {
        "nodes": list(analyzer.causal_graph.nodes),
        "edges": [{"source": u, "target": v} 
                 for u, v in analyzer.causal_graph.edges]
    }