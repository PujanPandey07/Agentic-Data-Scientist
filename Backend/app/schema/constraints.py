# schema/constraints.py
from pydantic import BaseModel, Field
from typing import Literal


class ConstraintItem(BaseModel):
    stage: Literal[
        "cleaning", "eda", "visualization", "feature_engineering",
        "model_selection", "hyperparameter_tuning", "evaluation", "reporting"
    ]
    instruction: str = Field(
        description="The explicit constraint, restated clearly")


class UserConstraints(BaseModel):
    constraints: list[ConstraintItem] = Field(
        default_factory=list,
        description="Explicit, non-negotiable instructions the user stated. "
                    "Do NOT include vague preferences or things left to judgment."
    )


