import asyncio
from graphs.workflow import builder
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from logging_config import setup_logging
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.types import Command
import aiosqlite


setup_logging()


def _prompt_for_decision(payload: dict) -> dict:
    """Show an interrupt payload and collect a decision dict shaped as
    {"approved": bool, "edit_instruction": str | None} — the shape both
    plan_review_node and confirm_refinement_node now expect back."""
    interrupt_type = payload.get("type", "unknown")

    print(f"\n========== INTERRUPT: {interrupt_type} ==========")
    print(payload.get("summary", ""))

    if interrupt_type == "plan_review":
        print(f"Tasks: {payload.get('tasks')}")
        print(f"Target column: {payload.get('target_column')}")
        print(f"Problem type: {payload.get('problem_type')}")
        constraints = payload.get("constraints") or {}
        if constraints:
            print("Constraints:")
            for stage, instrs in constraints.items():
                for instr in instrs:
                    print(f"  [{stage}] {instr}")

    elif interrupt_type == "refinement":
        print(f"Stage: {payload.get('stage')}")
        print(f"Instruction: {payload.get('instruction')}")
        print(f"Confidence: {payload.get('confidence')}")

    print("===========================================")

    choice = input("Approve? (y = yes / n = no / e = edit): ").strip().lower()

    if choice == "y":
        return {"approved": True, "edit_instruction": None}

    if choice == "e":
        edit_text = input("Enter your edit/instruction: ").strip()
        # plan_review treats any edit as "not approved yet, re-plan with this"
        # confirm_refinement treats an edit as "approved, but with this tweak"
        approved = interrupt_type == "refinement"
        return {"approved": approved, "edit_instruction": edit_text}

    return {"approved": False, "edit_instruction": None}


async def main():
    # ---------------------------------------------------------------
    # Pick which query to test by uncommenting one of these:
    # ---------------------------------------------------------------
    user_query = "Build a classification model on the Iris dataset using XGBoost"
    # user_query = "What does F1 score mean?"
    # user_query = "use XGBoost as the primary algorithm and tune hyperparameters"

    dataset_id = "08f054c7-0d54-4db4-95db-b84616f7bf25"

    # -----------------------------------------------------------------
    # True only on the very FIRST run for a given dataset_id/thread_id —
    # sends a full initial_state that OVERWRITES the checkpoint.
    # False for any follow-up call on an EXISTING thread (refinements,
    # explain_result, general questions) — sends only user_query +
    # dataset_id, leaving everything else as the checkpoint already has it.
    # -----------------------------------------------------------------
    is_first_run = True

    if is_first_run:
        graph_input = {
            "user_query": user_query,
            "dataset_id": dataset_id,
            "dataframe": None,
            "dataset_summary": None,
            "analysis_plan": None,
            "cleaning_report": None,
            "eda_report": None,
            "visualization_plan": None,
            "visualization_results": None,
            "feature_engineering_plan": None,
            "feature_engineering_report": None,
            "current_task": None,
            "remaining_tasks": [],
            "completed_tasks": [],
            "execution_logs": [],
            "intent": None,
            "direct_answer": None,
        }
    else:
        graph_input = {
            "user_query": user_query,
            "dataset_id": dataset_id,
        }

    print("\n========== RUNNING FULL GRAPH ==========\n")

    async with aiosqlite.connect("checkpoints.sqlite") as conn:
        checkpointer = AsyncSqliteSaver(
            conn,
            serde=JsonPlusSerializer(pickle_fallback=True),
        )
        graph = builder.compile(checkpointer=checkpointer)

        config = {"configurable": {"thread_id": dataset_id}}

        result = await graph.ainvoke(graph_input, config=config)

        # Loop: keep resuming as long as the graph keeps pausing.
        # plan_review_node can interrupt multiple times in a row (each
        # "edit" loops back for another review of the revised plan), and
        # confirm_refinement_node interrupts once for refine_step — this
        # loop handles either, and any sequence of them, generically.
        while "__interrupt__" in result:
            payload = result["__interrupt__"][0].value
            decision = _prompt_for_decision(payload)
            result = await graph.ainvoke(Command(resume=decision), config=config)

        final_state = result

    print("\n========== FINAL STATE ==========\n")

    print(f"INTENT: {final_state.get('intent')}")

    if final_state.get("plan_review_cancelled"):
        print("\nPlan was rejected — nothing executed.")
        print("\n========== DONE ==========\n")
        return

    if final_state.get("direct_answer"):
        print(f"\nDIRECT ANSWER:\n{final_state['direct_answer']}")
        print("\n========== DONE ==========\n")
        return

    if final_state.get("intent") == "refine_step" and not final_state.get("refine_confirmed"):
        print(f"\nREFINE TARGET: {final_state.get('refine_target')}")
        print(f"INSTRUCTION: {final_state.get('refine_instruction')}")
        print(f"CONFIDENCE: {final_state.get('refine_confidence')}")
        print("\n(Not confirmed — nothing executed)")
        print("\n========== DONE ==========\n")
        return

    # Otherwise — a full run_pipeline execution, or a CONFIRMED refine_step
    # that cascaded through real stages. Print everything as before.
    print(f"COMPLETED TASKS: {final_state['completed_tasks']}")
    print(f"CURRENT TASK: {final_state['current_task']}")
    print(f"REMAINING TASKS: {final_state['remaining_tasks']}")

    print(
        f"\nDATAFRAME SHAPE: {final_state['dataframe'].shape if final_state['dataframe'] is not None else 'None'}")

    if final_state.get("cleaning_report"):
        print(
            f"\nCLEANING: {final_state['cleaning_report']['original_rows']} -> {final_state['cleaning_report']['final_rows']} rows")

    if final_state.get("eda_report"):
        print(f"\nEDA: {final_state['eda_report'].get('shape')}")

    if final_state.get("visualization_results"):
        print(
            f"\nVISUALIZATIONS: {len(final_state['visualization_results'])} charts generated")
        for v in final_state["visualization_results"]:
            print(f"  - {v['chart_type']}: {v.get('path', 'N/A')}")

    if final_state.get("feature_engineering_plan"):
        plan = final_state["feature_engineering_plan"]
        print(f"\nFE STRATEGY: {plan.strategy}")
        print(f"FE STEPS: {len(plan.steps)}")
        for step in plan.steps:
            print(f"  - {step.action}: {step.columns}")

    if final_state.get("feature_engineering_report"):
        report = final_state["feature_engineering_report"]
        if "error" in report:
            print(f"\nFE ERROR: {report['error']}")
        else:
            print(
                f"\nFE RESULT: train {report.get('final_train_shape', 'N/A')}, "
                f"test {report.get('final_test_shape', 'N/A')}")
            print(
                f"FE STATUS: {report.get('steps_executed')} succeeded, {report.get('steps_failed')} failed")

    if final_state.get("training_report"):
        report = final_state["training_report"]
        if "error" in report:
            print(f"\nTRAINING ERROR: {report['error']}")
        else:
            print(f"\nTRAINING:")
            print(f"  Strategy: {report['strategy']}")
            print(f"  Best algorithm: {report['best_algorithm']}")
            print(f"  Best CV score: {report['best_mean_cv_score']}")
            print(f"  Model path: {report.get('model_path', 'N/A')}")
            print(f"  Time spent: {report['time_spent_seconds']}s")
            print(f"  Candidates tried:")
            for r in report["candidates_results"]:
                status = "✅" if r["status"] == "success" else "❌"
                if r["status"] == "success":
                    print(
                        f"    {status} {r['algorithm']}: {r['mean_cv_score']:.5f} (+/- {r['std_cv_score']:.5f}) [{r['actual_time_seconds']}s]")
                else:
                    print(
                        f"    {status} {r['algorithm']}: FAILED - {r.get('error', 'unknown')}")

    if final_state.get("hyperparameter_tuning_report"):
        report = final_state["hyperparameter_tuning_report"]
        print(f"\nHYPERPARAMETER TUNING:")
        print(f"  Status: {report.get('status', 'completed')}")
        if report.get('status') != 'skipped':
            print(f"  Best trial score: {report.get('best_trial_score')}")
            print(
                f"  Trials completed: {report.get('num_trials_completed')}")
            print(f"  Best params: {report.get('best_params')}")
            print(
                f"  Top important param: {max(report.get('param_importance', {}), key=report.get('param_importance', {}).get) if report.get('param_importance') else 'N/A'}")
            print(f"  Tuned model: {report.get('tuned_model_path')}")

    if final_state.get("trained_model_path"):
        print(f"\nMODEL SAVED: {final_state['trained_model_path']}")

        if final_state.get("evaluation_report"):
            report = final_state["evaluation_report"]
            if "error" in report:
                print(f"\nEVALUATION ERROR: {report['error']}")
            else:
                print(f"\nEVALUATION:")
                print(f"  Problem type: {report['problem_type']}")
                print(f"  Samples evaluated: {report['num_samples']}")
                print(f"  Features used: {report['num_features']}")

                metrics = report["metrics"]
                if report["problem_type"] == "classification":
                    print(f"  Accuracy: {metrics['accuracy']}")
                    print(f"  F1 (weighted): {metrics['f1_weighted']}")
                    print(f"  F1 (macro): {metrics['f1_macro']}")
                    print(
                        f"  Precision (weighted): {metrics['precision_weighted']}")
                    print(f"  Recall (weighted): {metrics['recall_weighted']}")
                else:
                    print(f"  RMSE: {metrics['rmse']}")
                    print(f"  R²: {metrics['r2']}")
                    print(f"  MAE: {metrics['mae']}")

                artifacts = report.get("artifacts", {})
                if "confusion_matrix_path" in artifacts:
                    print(
                        f"  Confusion matrix: {artifacts['confusion_matrix_path']}")
                if "residual_plot_path" in artifacts:
                    print(
                        f"  Residual plot: {artifacts['residual_plot_path']}")

    if final_state.get("final_report"):
        report = final_state["final_report"]
        print(f"\nFINAL REPORT:")
        print(f"  Saved to: {report.get('report_path')}")
        print(f"  Problem type: {report['problem_type']}")
        print(f"  Best model: {report['conclusions']['best_model']}")
        print(f"  CV score: {report['conclusions']['cv_score']}")
        print(
            f"  Final accuracy: {report['conclusions'].get('final_accuracy')}")
        print(f"  Recommendation: {report['conclusions']['recommendation']}")

    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    asyncio.run(main())
