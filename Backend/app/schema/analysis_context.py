from typing import Any

from pydantic import BaseModel, Field


class AnalysisContext(BaseModel):
    """Compact, LLM-facing representation of a completed analysis."""

    dataset_id: str
    generated_at: str
    dataset: dict[str, Any] = Field(default_factory=dict)
    cleaning: dict[str, Any] | None = None
    eda: dict[str, Any] | None = None
    visualization: dict[str, Any] = Field(default_factory=dict)
    feature_engineering: dict[str, Any] | None = None
    model_selection: dict[str, Any] | None = None
    training: dict[str, Any] | None = None
    hyperparameter_tuning: dict[str, Any] | None = None
    evaluation: dict[str, Any] | None = None
    conclusions: dict[str, Any] = Field(default_factory=dict)
