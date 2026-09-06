import logging

logger = logging.getLogger(__name__)


class ExecutionService:

    def advance_task(
        self,
        state,
        completed_task,
        message,
        status="success",
    ):
        # Only count it as "completed" if it actually ran successfully.
        # Skipped/errored stages are logged but not added to completed_tasks,
        # so history doesn't lie about what actually executed.
        if status == "success":
            state["completed_tasks"].append(completed_task)

        if state["remaining_tasks"]:
            state["current_task"] = state["remaining_tasks"].pop(0)
        else:
            state["current_task"] = None

        state["execution_logs"].append(
            {
                "node": completed_task,
                "status": status,
                "message": message,
            }
        )

        logger.info(
            f"Task '{completed_task}' finished with status='{status}'. Next task: {state['current_task']}"
        )

        return state


execution_service = ExecutionService()
