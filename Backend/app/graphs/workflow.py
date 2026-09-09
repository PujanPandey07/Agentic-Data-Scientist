from langgraph.graph import START, END, StateGraph

from graphs.nodes import (
    confirm_refinement_node, dataset_node, planner_node, initialize_execution_node, advance_execution, refinement_cancelled_node, route_after_confirmation,
    router, cleaning_node, eda_node, visualization_node, feature_engineering_node,
    training_node, evaluation_node, reporting_node, route_task, planner_agent,
    visualization_planner_node, feature_engineering_planner_node, feature_engineering_node,
    model_selection_planner_node, training_node, evaluation_node, reporting_node,
    hyperparameter_tuning_node,
    intent_router_node, route_intent, direct_answer_node,  refine_target_node, extract_constraints_node,
    plan_review_node, plan_review_cancelled_node, route_after_plan_review,
)

from graphs.state import GraphState


builder = StateGraph(GraphState)

# Register nodes
builder.add_node("dataset", dataset_node)
builder.add_node("planner", planner_node)
builder.add_node("initialize_execution", initialize_execution_node)
builder.add_node("cleaning", cleaning_node)
builder.add_node("eda", eda_node)
builder.add_node("visualization", visualization_node)
builder.add_node("model_selection_planner", model_selection_planner_node)
builder.add_node("training", training_node)
builder.add_node("evaluation", evaluation_node)
builder.add_node("reporting", reporting_node)
builder.add_node("router", router)
builder.add_node("visualization_planner", visualization_planner_node)
builder.add_node("feature_engineering_planner",
                 feature_engineering_planner_node)
builder.add_node("feature_engineering", feature_engineering_node)
builder.add_node("hyperparameter_tuning", hyperparameter_tuning_node)
builder.add_node("intent_router", intent_router_node)
builder.add_node("direct_answer", direct_answer_node)
builder.add_node("refine_target", refine_target_node)
builder.add_node("confirm_refinement", confirm_refinement_node)
builder.add_node("refinement_cancelled", refinement_cancelled_node)
builder.add_node("extract_constraints", extract_constraints_node)
builder.add_node("plan_review", plan_review_node)
builder.add_node("plan_review_cancelled", plan_review_cancelled_node)

# Define workflow
builder.add_edge(START, "intent_router")
builder.add_conditional_edges(
    "intent_router",
    route_intent,
    {
        "run_pipeline": "dataset",
        "direct_answer": "direct_answer",
        "refine_step": "refine_target",
    },
)
builder.add_edge("direct_answer", END)
builder.add_conditional_edges(
    "confirm_refinement",
    route_after_confirmation,
    {
        "router": "router",
        "cancelled": "refinement_cancelled",
    },
)


builder.add_edge("dataset", "planner")
builder.add_edge("planner", "extract_constraints")

# extract_constraints now goes to plan_review instead of straight to
# initialize_execution — the whole proposed plan is reviewed once, up
# front, before any stage runs.
builder.add_edge("extract_constraints", "plan_review")
builder.add_conditional_edges(
    "plan_review",
    route_after_plan_review,
    {
        "proceed": "initialize_execution",
        "revise": "plan_review",   # loop back to review the just-updated plan
        "cancelled": "plan_review_cancelled",
    },
)
builder.add_edge("plan_review_cancelled", END)

builder.add_edge("initialize_execution", "router")
builder.add_conditional_edges(
    "router",
    route_task,
    {
        "cleaning": "cleaning",
        "eda": "eda",
        "visualization": "visualization_planner",
        "feature_engineering": "feature_engineering_planner",
        "model_selection": "model_selection_planner",
        "hyperparameter_tuning": "hyperparameter_tuning",
        "evaluation": "evaluation",
        "reporting": "reporting",
        "visualization_planner": "visualization_planner",
        "feature_engineering_planner": "feature_engineering_planner",
        END: END,
    },
)


builder.add_edge("cleaning", "router")
builder.add_edge("eda", "router")

builder.add_edge("visualization_planner", "visualization")
builder.add_edge("visualization", "router")

builder.add_edge("feature_engineering_planner",
                 "feature_engineering")
builder.add_edge("feature_engineering", "router")
builder.add_edge("model_selection_planner", "training")
builder.add_edge("training", "router")
builder.add_edge("hyperparameter_tuning", "router")
builder.add_edge("evaluation", "router")
builder.add_edge("reporting", "router")


builder.add_edge("refine_target", "confirm_refinement")
builder.add_edge("refinement_cancelled", END)
