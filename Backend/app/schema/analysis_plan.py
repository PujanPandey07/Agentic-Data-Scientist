from pydantic import BaseModel


class AnalysisPlan(BaseModel):
    user_intent: str
    problem_type: str | None = None
    target_column: str | None = None
    tasks: list[str]
    reasoning: str
