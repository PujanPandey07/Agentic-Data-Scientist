from pydantic import BaseModel, Field
from typing import Literal


class IntentClassification(BaseModel):
    intent: Literal["run_pipeline", "explain_result", "general_question"] = Field(
        description="What kind of request this is"
    )
    reasoning: str = Field(
        description="Brief reasoning for why this intent was chosen"
    )
