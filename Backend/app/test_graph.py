import asyncio
# or however you import your compiled graph
from graphs.workflow import graph


async def main():
    # Initial state
    initial_state = {
        "user_query": "Build a classification model on the Iris dataset",
        "dataset_id": "08f054c7-0d54-4db4-95db-b84616f7bf25",
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
    }

    print("\n========== RUNNING FULL GRAPH ==========\n")

    # Run the graph
    final_state = await graph.ainvoke(initial_state)

    print("\n========== FINAL STATE ==========\n")

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
        print(
            f"\nFE RESULT: {report['original_shape']} -> {report['final_shape']}")
        print(
            f"FE STATUS: {report['steps_executed']} succeeded, {report['steps_failed']} failed")

    if final_state.get("training_report"):
        report = final_state["training_report"]
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

    if final_state.get("trained_model_path"):
        print(f"\nMODEL SAVED: {final_state['trained_model_path']}")

        if final_state.get("evaluation_report"):
            report = final_state["evaluation_report"]
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
                print(f"  Residual plot: {artifacts['residual_plot_path']}")

    print("\n========== DONE ==========\n")


if __name__ == "__main__":
    asyncio.run(main())
