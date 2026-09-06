from pydantic import BaseModel, Field
from typing import Literal


class RefineTarget(BaseModel):
    target_stage: Literal[
        "cleaning", "eda", "visualization", "feature_engineering",
        "model_selection", "hyperparameter_tuning", "evaluation", "reporting"
    ] = Field(description="Which pipeline stage this refinement request refers to")
    instruction: str = Field(
        description="Clear restatement of what the user wants changed, to hand to that stage's planner")
    confidence: float = Field(
        description="0.0-1.0 confidence in the target_stage classification")
