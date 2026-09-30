# prompts/time_series.py

TIME_SERIES_ANALYSIS_PROMPT = """You are an expert time-series data scientist.

Given a dataset summary, your job is to determine the best time-series analysis approach.

Respond with a JSON object matching this schema:
{
  "time_column": "<column name that holds the timestamp/date>",
  "target_column": "<column name to forecast>",
  "frequency": "<one of: hourly, daily, weekly, monthly>",
  "horizon": <integer — how many steps ahead to forecast, default 10>,
  "seasonality_detected": <true/false based on the summary>,
  "trend_detected": <true/false>,
  "is_stationary": <true/false — initial guess from the data description>,
  "recommended_models": ["xgboost_lags"],
  "lag_features": [1, 2, 3, 7],
  "rolling_windows": [3, 7]
}

Rules:
- time_column must be an actual column in the dataset (pick the most likely datetime column)
- target_column must be a numeric column the user wants to forecast
- Prefer lag features matching the frequency: hourly→[1,2,3,24], daily→[1,2,3,7], weekly→[1,2,4,8]
- Always include xgboost_lags in recommended_models — it works well as a universal baseline
- Only output the JSON object, nothing else
"""

TIME_SERIES_MODEL_SELECTION_PROMPT = """You are a time-series forecasting expert.

Given the analysis results (stationarity test, decomposition), select the best
forecasting approach and configuration.

Consider:
- If stationary and no clear trend/seasonality: ARIMA or simple lag-based XGBoost
- If non-stationary with trend: differencing + XGBoost lags
- If strong seasonality: XGBoost with seasonal lag features (e.g. lag_7 for daily data)
- For most cases, XGBoost with carefully chosen lags is the safest choice

Output a JSON object:
{
  "primary_model": "xgboost_lags",
  "lag_features": [1, 2, 3, 7],
  "rolling_windows": [3, 7],
  "n_estimators": 200,
  "reasoning": "<brief explanation of your choice>"
}
"""
