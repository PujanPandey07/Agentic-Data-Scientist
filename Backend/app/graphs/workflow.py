from langgraph.graph import START, END, StateGraph

from graphs.nodes import (
    confirm_refinement_node, dataset_node, planner_node, initialize_execution_node,
    advance_execution, refinement_cancelled_node, resolve_llm_node, route_after_confirmation,
    router, cleaning_node, eda_node, visualization_node,
    visualization_planner_node, feature_engineering_planner_node, feature_engineering_node,
    model_selection_planner_node, enqueue_training_node, poll_training_node,
    enqueue_tuning_node, poll_tuning_node,
    evaluation_node, reporting_node, route_task, planner_agent,
    intent_router_node, route_intent, direct_answer_node, refine_target_node, extract_constraints_node,
    plan_review_node, plan_review_cancelled_node, route_after_plan_review,
)

from graphs.state import GraphState


def _add_shared_nodes_and_entry_edges(builder: StateGraph) -> None:
    """Everything that doesn't care whether this run is supervised or
    unsupervised: entry routing, dataset/planning (used only on the
    dataset->planner path — i.e. when analysis_plan was NOT already built
    outside the graph), plan review, cleaning/EDA/visualization, the
    router itself, and refinement. Shared so both graph builders wire the
    exact same node objects rather than duplicating logic."""
    builder.add_node("resolve_llm", resolve_llm_node)
    builder.add_node("intent_router", intent_router_node)
    builder.add_node("direct_answer", direct_answer_node)

    builder.add_node("dataset", dataset_node)
    builder.add_node("planner", planner_node)
    builder.add_node("extract_constraints", extract_constraints_node)
    builder.add_node("plan_review", plan_review_node)
    builder.add_node("plan_review_cancelled", plan_review_cancelled_node)

    builder.add_node("initialize_execution", initialize_execution_node)
    builder.add_node("router", router)

    builder.add_node("cleaning", cleaning_node)
    builder.add_node("eda", eda_node)
    builder.add_node("visualization_planner", visualization_planner_node)
    builder.add_node("visualization", visualization_node)

    builder.add_node("refine_target", refine_target_node)
    builder.add_node("confirm_refinement", confirm_refinement_node)
    builder.add_node("refinement_cancelled", refinement_cancelled_node)

    # --- entry ---
    builder.add_edge(START, "resolve_llm")
    builder.add_edge("resolve_llm", "intent_router")
    builder.add_conditional_edges(
        "intent_router",
        route_intent,
        {
            "run_pipeline": "dataset",
            # NEW: analysis_plan was already built in api/runs.py before
            # this graph was even chosen (needed to know problem_type up
            # front) — skip dataset/planner, go straight to constraints.
            "extract_constraints": "extract_constraints",
            "resume_pipeline": "router",
            "direct_answer": "direct_answer",
            "refine_step": "refine_target",
        },
    )
    builder.add_edge("direct_answer", END)

    # --- planning path (only reached when analysis_plan wasn't pre-built) ---
    builder.add_edge("dataset", "planner")
    builder.add_edge("planner", "extract_constraints")
    builder.add_edge("extract_constraints", "plan_review")
    builder.add_conditional_edges(
        "plan_review",
        route_after_plan_review,
        {
            "proceed": "initialize_execution",
            "revise": "plan_review",
            "cancelled": "plan_review_cancelled",
        },
    )
    builder.add_edge("plan_review_cancelled", END)
    builder.add_edge("initialize_execution", "router")

    # --- stages shared by every family ---
    builder.add_edge("cleaning", "router")
    builder.add_edge("eda", "router")
    builder.add_edge("visualization_planner", "visualization")
    builder.add_edge("visualization", "router")

    # --- refinement ---
    builder.add_edge("refine_target", "confirm_refinement")
    builder.add_conditional_edges(
        "confirm_refinement",
        route_after_confirmation,
        {
            "router": "router",
            "cancelled": "refinement_cancelled",
        },
    )
    builder.add_edge("refinement_cancelled", END)


def build_supervised_graph() -> StateGraph:
    """Classification/regression pipeline — identical topology to the
    original single-graph setup."""
    builder = StateGraph(GraphState)
    _add_shared_nodes_and_entry_edges(builder)

    builder.add_node("feature_engineering_planner",
                     feature_engineering_planner_node)
    builder.add_node("feature_engineering", feature_engineering_node)
    builder.add_node("model_selection_planner", model_selection_planner_node)
    builder.add_node("enqueue_training", enqueue_training_node)
    builder.add_node("poll_training", poll_training_node)
    builder.add_node("enqueue_tuning", enqueue_tuning_node)
    builder.add_node("poll_tuning", poll_tuning_node)
    builder.add_node("evaluation", evaluation_node)
    builder.add_node("reporting", reporting_node)

    builder.add_conditional_edges(
        "router",
        route_task,
        {
            "cleaning": "cleaning",
            "eda": "eda",
            "visualization": "visualization_planner",
            "feature_engineering": "feature_engineering_planner",
            "model_selection": "model_selection_planner",
            "hyperparameter_tuning": "enqueue_tuning",
            "evaluation": "evaluation",
            "reporting": "reporting",
            "visualization_planner": "visualization_planner",
            "feature_engineering_planner": "feature_engineering_planner",
            END: END,
        },
    )

    builder.add_edge("feature_engineering_planner", "feature_engineering")
    builder.add_edge("feature_engineering", "router")
    builder.add_edge("model_selection_planner", "enqueue_training")
    builder.add_edge("enqueue_training", "poll_training")
    builder.add_edge("poll_training", "router")
    builder.add_edge("enqueue_tuning", "poll_tuning")
    builder.add_edge("poll_tuning", "router")
    builder.add_edge("evaluation", "router")
    builder.add_edge("reporting", "router")

    return builder


def build_unsupervised_graph() -> StateGraph:
    """PLACEHOLDER — clustering nodes (clustering_model_selection_planner,
    enqueue/poll_clustering, clustering_evaluation) don't exist yet, so
    this currently duplicates the supervised topology exactly. This is
    intentional and safe to ship now: nothing in the planner can yet
    produce problem_type="clustering" (that's the next schema change), so
    this graph is unreachable in practice until that lands. It exists now
    purely so the dispatch mechanism (api/runs.py choosing a graph,
    api/chat.py looking up pipeline_family) is wired end-to-end and
    tested before the clustering-specific nodes are written. REPLACE the
    body of this function once those nodes exist — do not leave it as a
    supervised-graph duplicate long-term.
    """
    return build_supervised_graph()


# Compiled once at import time is intentionally NOT done here — main.py's
# lifespan/startup is expected to call these builders and .compile(...)
# each with the shared checkpointer, then assign to app.state.graph and
# app.state.unsupervised_graph. See the note left for main.py.
