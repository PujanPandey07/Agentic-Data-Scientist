from langgraph.graph import START, END, StateGraph

from graphs.nodes import dataset_node, planner_node, initialize_execution_node, advance_execution, router, cleaning_node, eda_node, visualization_node, feature_engineering_node, training_node, evaluation_node, reporting_node, route_task, planner_agent

from graphs.state import GraphState

builder = StateGraph(GraphState)

# Register nodes
builder.add_node("dataset", dataset_node)
builder.add_node("planner", planner_node)
builder.add_node("initialize_execution", initialize_execution_node)
builder.add_node("cleaning", cleaning_node)
builder.add_node("eda", eda_node)
builder.add_node("visualization", visualization_node)
builder.add_node("feature_engineering", feature_engineering_node)
builder.add_node("training", training_node)
builder.add_node("evaluation", evaluation_node)
builder.add_node("reporting", reporting_node)
builder.add_node("router", router)


# Define workflow
builder.add_edge(START, "dataset")
builder.add_edge("dataset", "planner")
builder.add_edge("planner", "initialize_execution")
builder.add_edge("initialize_execution", "router")
builder.add_conditional_edges(
    "router",
    route_task,
    {
        "cleaning": "cleaning",
        "eda": "eda",
        "visualization": "visualization",
        "feature_engineering": "feature_engineering",
        "training": "training",
        "evaluation": "evaluation",
        "reporting": "reporting",
        END: END,
    },

)
builder.add_edge("cleaning", "router")
builder.add_edge("eda", "router")
builder.add_edge("visualization", "router")
builder.add_edge("feature_engineering", "router")
builder.add_edge("training", "router")
builder.add_edge("evaluation", "router")
builder.add_edge("reporting", "router")


# Compile graph
graph = builder.compile()
