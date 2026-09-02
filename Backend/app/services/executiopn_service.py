import logging

logger = logging.getLogger(__name__)


class ExecutionService:

    def advance_task(
        self,
        state,
        completed_task,
        message
    ):

        state["completed_tasks"].append(completed_task)

        if state["remaining_tasks"]:
            state["current_task"] = state["remaining_tasks"].pop(0)
        else:
            state["current_task"] = None

        state["execution_logs"].append(
            {
                "node": completed_task,
                "status": "success",
                "message": message
            }
        )

        logger.info(
            f"Task '{completed_task}' completed. Next task: {state['current_task']}"
        )

        return state


execution_service = ExecutionService()
