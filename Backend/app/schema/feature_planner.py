from pydantic import BaseModel, Field
from typing import Literal, Optional


class FeatureEngineeringStep(BaseModel):
    action: Literal[
        "drop_columns",
        "one_hot_encode",
        "label_encode",
        "standard_scale",
        "minmax_scale",
        "robust_scale",
        "log_transform",
        "sqrt_transform",
        "power_transform",
        "binning",
        "create_interaction",
        "create_polynomial",
        "drop_low_variance",
        "drop_high_correlation",
    ] = Field(description="The feature engineering action to perform")

    columns: list[str] = Field(
        description="List of column names to apply the action to"
    )

    params: Optional[dict] = Field(
        default=None,
        description="Action-specific parameters. Examples: {'degree': 2} for polynomial, "
                    "{'bins': 5, 'strategy': 'quantile'} for binning, "
                    "{'operation': 'multiply'} for interaction, "
                    "{'threshold': 0.9} for correlation drop, "
                    "{'drop_first': true} for one-hot.",
    )

    reason: str = Field(
        description="Clear explanation for why this step is needed, citing EDA evidence"
    )


class FeatureEngineeringPlan(BaseModel):

    strategy: Literal["conservative", "balanced", "aggressive"] = Field(
        default="balanced",
        description="The feature engineering strategy to use"
    )
    steps: list[FeatureEngineeringStep] = Field(
        description="Ordered list of feature engineering steps. Order matters: "
                    "drop columns before scaling, create features before encoding, etc."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Warnings about potential data leakage or post-split requirements"
    )
    notes: list[str] = Field(
        default_factory=list,
        description="Additional recommendations (e.g., 'Consider target encoding after split')"
    )
