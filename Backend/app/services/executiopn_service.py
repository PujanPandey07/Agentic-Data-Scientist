from graphs.state import GraphState


class ExecutionService:

    def advance_task(
        self,
        state: GraphState,
        completed_task: str,
        message: str,
    ):

        state["completed_tasks"].append(completed_task)

        node = state["current_task"]

        if state["remaining_tasks"]:
            state["current_task"] = state["remaining_tasks"].pop(0)
        else:
            state["current_task"] = None

        state["execution_logs"].append(
            {
                "node": node,
                "status": "success",
                "message": message,
            }
        )

        return state


execution_service = ExecutionService()
