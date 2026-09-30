# services/time_series_service.py
"""
Core time-series logic: stationarity testing, decomposition,
lag-feature engineering, chronological splitting, and model fitting.

All third-party imports are guarded with try/except so the service
degrades gracefully when statsmodels or xgboost are absent.
"""
from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── optional imports ────────────────────────────────────────────────────────

try:
    from statsmodels.tsa.stattools import adfuller
    _HAS_STATSMODELS = True
except ImportError:
    _HAS_STATSMODELS = False
    logger.warning("statsmodels not installed — stationarity will use a heuristic fallback")

try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    _HAS_DECOMPOSE = True
except ImportError:
    _HAS_DECOMPOSE = False

try:
    import xgboost as xgb
    _HAS_XGB = True
except ImportError:
    _HAS_XGB = False
    logger.warning("xgboost not installed — falling back to sklearn GradientBoosting")


# ── helper: safe MAPE ───────────────────────────────────────────────────────

def _mape(actual: np.ndarray, predicted: np.ndarray) -> float:
    mask = actual != 0
    if not mask.any():
        return float("nan")
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


# ── main service ────────────────────────────────────────────────────────────

class TimeSeriesService:
    """Stateless helper — all methods are pure functions on their inputs."""

    # ── 1. detect time column ──────────────────────────────────────────────

    def detect_time_column(self, df: pd.DataFrame) -> str | None:
        """Return the first column that looks like a datetime index."""
        # Priority: columns already parsed as datetime dtype
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                return col

        # Fallback: name heuristics
        hints = {"date", "time", "timestamp", "period", "datetime",
                 "year", "month", "week", "day", "hour"}
        for col in df.columns:
            if any(h in col.lower() for h in hints):
                try:
                    pd.to_datetime(df[col], infer_datetime_format=True)
                    return col
                except Exception:
                    pass

        return None

    # ── 2. stationarity ────────────────────────────────────────────────────

    def make_stationary(
        self, series: pd.Series, max_diffs: int = 2
    ) -> tuple[pd.Series, bool, int]:
        """
        Apply differencing until the ADF test passes or max_diffs reached.
        Returns (transformed_series, is_stationary, n_diffs).
        """
        s = series.dropna()
        n_diffs = 0

        for _ in range(max_diffs + 1):
            stationary = self._is_stationary(s)
            if stationary:
                return s, True, n_diffs
            if n_diffs < max_diffs:
                s = s.diff().dropna()
                n_diffs += 1

        return s, False, n_diffs

    def _is_stationary(self, series: pd.Series) -> bool:
        if _HAS_STATSMODELS and len(series) > 20:
            try:
                p_value = adfuller(series, autolag="AIC")[1]
                return float(p_value) < 0.05
            except Exception:
                pass
        # Heuristic: compare variance of first vs second half
        mid = len(series) // 2
        var_ratio = series.iloc[:mid].var() / (series.iloc[mid:].var() + 1e-9)
        return 0.5 < var_ratio < 2.0

    # ── 3. decomposition ───────────────────────────────────────────────────

    def decompose(
        self,
        df: pd.DataFrame,
        time_col: str,
        target_col: str,
        frequency: str = "daily",
    ) -> dict[str, Any]:
        """
        Seasonal decomposition → returns trend/seasonal/residual as lists.
        Falls back to a simple rolling-mean trend when statsmodels is absent.
        """
        freq_map = {"hourly": 24, "daily": 7, "weekly": 52, "monthly": 12}
        period = freq_map.get(frequency, 7)

        try:
            ts = df.set_index(pd.to_datetime(df[time_col]))[target_col].sort_index()
        except Exception as exc:
            return {"error": str(exc)}

        if _HAS_DECOMPOSE and len(ts) >= period * 2:
            try:
                result = seasonal_decompose(ts, model="additive", period=period, extrapolate_trend="freq")
                trend_vals = result.trend.fillna(method="bfill").fillna(method="ffill").tolist()
                seasonal_vals = result.seasonal.tolist()
                residual_vals = result.resid.fillna(0).tolist()
                trend_detected = float(np.std(trend_vals)) > 0.01 * float(np.std(ts))
                seasonal_strength = float(np.std(seasonal_vals)) / (float(np.std(ts)) + 1e-9)
                return {
                    "trend": trend_vals[:50],        # truncate for state storage
                    "seasonal": seasonal_vals[:50],
                    "residual": residual_vals[:50],
                    "trend_detected": trend_detected,
                    "seasonality_detected": seasonal_strength > 0.1,
                    "period": period,
                }
            except Exception as exc:
                logger.warning("Decomposition failed: %s", exc)

        # Fallback: simple rolling mean trend
        window = min(period, len(ts) // 4 or 1)
        trend = ts.rolling(window=window, center=True).mean().fillna(ts)
        return {
            "trend": trend.tolist()[:50],
            "seasonal": [],
            "residual": (ts - trend).tolist()[:50],
            "trend_detected": float(trend.iloc[-1]) > float(trend.iloc[0]),
            "seasonality_detected": False,
            "period": period,
        }

    # ── 4. lag features ────────────────────────────────────────────────────

    def create_lag_features(
        self,
        df: pd.DataFrame,
        target_col: str,
        lags: list[int],
        rolling_windows: list[int],
        time_col: str | None = None,
    ) -> pd.DataFrame:
        """Add lag and rolling-mean columns; optionally extract date parts."""
        out = df.copy()

        for lag in lags:
            out[f"lag_{lag}"] = out[target_col].shift(lag)

        for w in rolling_windows:
            out[f"rolling_mean_{w}"] = (
                out[target_col].shift(1).rolling(window=w).mean()
            )

        if time_col and time_col in out.columns:
            try:
                dt = pd.to_datetime(out[time_col])
                out["_ts_dayofweek"] = dt.dt.dayofweek
                out["_ts_month"] = dt.dt.month
                out["_ts_quarter"] = dt.dt.quarter
            except Exception:
                pass

        return out.dropna()

    # ── 5. chronological split ─────────────────────────────────────────────

    def chronological_split(
        self,
        df: pd.DataFrame,
        test_ratio: float = 0.20,
        time_col: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Split by position (never random). If time_col given, sort first.
        Last `test_ratio` fraction is the test set.
        """
        if time_col and time_col in df.columns:
            df = df.sort_values(time_col).reset_index(drop=True)

        n = len(df)
        split = max(1, int(n * (1 - test_ratio)))
        return df.iloc[:split].copy(), df.iloc[split:].copy()

    # ── 6. model fitting ───────────────────────────────────────────────────

    def fit_xgboost_lags(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        target_col: str,
        feature_cols: list[str],
    ) -> dict[str, Any]:
        """
        Fit XGBoost (or GradientBoosting fallback) on lag features.
        Returns MAE, RMSE, MAPE, and first 50 predictions.
        """
        X_train = train_df[feature_cols].values
        y_train = train_df[target_col].values
        X_test  = test_df[feature_cols].values
        y_test  = test_df[target_col].values

        try:
            if _HAS_XGB:
                model = xgb.XGBRegressor(
                    n_estimators=200,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    random_state=42,
                    verbosity=0,
                )
            else:
                from sklearn.ensemble import GradientBoostingRegressor
                model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=4, learning_rate=0.05, random_state=42
                )

            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            mae  = float(np.mean(np.abs(y_test - preds)))
            rmse = float(math.sqrt(np.mean((y_test - preds) ** 2)))
            mape = _mape(y_test, preds)

            # Feature importances (if available)
            importances = {}
            if hasattr(model, "feature_importances_"):
                for name, imp in zip(feature_cols, model.feature_importances_):
                    importances[name] = round(float(imp), 4)

            return {
                "model": "XGBoost" if _HAS_XGB else "GradientBoosting",
                "n_train": int(len(y_train)),
                "n_test":  int(len(y_test)),
                "metrics": {
                    "mae":  round(mae, 4),
                    "rmse": round(rmse, 4),
                    "mape": round(mape, 2) if not math.isnan(mape) else None,
                },
                "feature_importances": importances,
                "predictions": preds.tolist()[:50],
                "actuals":     y_test.tolist()[:50],
            }

        except Exception as exc:
            logger.exception("TS model fit failed: %s", exc)
            return {"error": str(exc)}

    # ── 7. select feature columns ──────────────────────────────────────────

    def get_feature_columns(
        self,
        df: pd.DataFrame,
        target_col: str,
        time_col: str | None,
    ) -> list[str]:
        """Return all lag/rolling/date-part columns usable as features."""
        exclude = {target_col}
        if time_col:
            exclude.add(time_col)
        return [
            c for c in df.columns
            if c not in exclude
            and (
                c.startswith("lag_")
                or c.startswith("rolling_")
                or c.startswith("_ts_")
            )
        ]


time_series_service = TimeSeriesService()
