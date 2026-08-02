from langgraph.graph import START, END, StateGraph

from graphs.nodes import dataset_node, planner_node
from graphs.state import GraphState

builder = StateGraph(GraphState)

# Register nodes
builder.add_node("dataset", dataset_node)
builder.add_node("planner", planner_node)

# Define workflow
builder.add_edge(START, "dataset")
builder.add_edge("dataset", "planner")
builder.add_edge("planner", END)

# Compile graph
graph = builder.compile()
