from typing import Literal, Optional
from pydantic import BaseModel


class AnalysisPlan(BaseModel):
    user_intent: str
    problem_type: Optional[Literal["classification", "regression", "clustering", "time_series"]] = None
    target_column: str | None = None
    tasks: list[str]
    reasoning: str
