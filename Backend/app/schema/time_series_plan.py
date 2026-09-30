# schema/time_series_plan.py
from pydantic import BaseModel, Field


class TimeSeriesPlan(BaseModel):
    time_column: str
    target_column: str
    frequency: str = "daily"          # "hourly", "daily", "weekly", "monthly"
    horizon: int = Field(default=10)  # steps to forecast
    seasonality_detected: bool = False
    trend_detected: bool = False
    is_stationary: bool = False
    recommended_models: list[str] = Field(
        default_factory=lambda: ["xgboost_lags"]
    )
    lag_features: list[int] = Field(default_factory=lambda: [1, 2, 3, 7])
    rolling_windows: list[int] = Field(default_factory=lambda: [3, 7])
