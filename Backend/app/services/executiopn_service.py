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

        return state


execution_service = ExecutionService()
