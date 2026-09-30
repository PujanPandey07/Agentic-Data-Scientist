# agents/time_series.py
import json
import logging

from llm.provider import get_llm
from prompts.time_series import TIME_SERIES_ANALYSIS_PROMPT, TIME_SERIES_MODEL_SELECTION_PROMPT
from schema.time_series_plan import TimeSeriesPlan

logger = logging.getLogger(__name__)


class TimeSeriesAgent:
    """LLM reasoning for time-series analysis and model selection.
    No self.llm at import time — fresh client per call for BYOK support."""

    async def analyze(
        self,
        user_query: str,
        dataset_summary,
        llm_config: dict | None = None,
    ) -> TimeSeriesPlan:
        """Produce a TimeSeriesPlan from the dataset summary."""
        logger.info("TS agent: analyzing dataset for time-series plan")

        llm = get_llm(llm_config)

        summary_text = (
            dataset_summary.model_dump_json(indent=2)
            if dataset_summary else "Not available"
        )

        messages = [
            {"role": "system", "content": TIME_SERIES_ANALYSIS_PROMPT},
            {"role": "user", "content": (
                f"User query:\n{user_query}\n\n"
                f"Dataset summary:\n{summary_text}\n\n"
                f"Produce the TimeSeriesPlan JSON now."
            )},
        ]

        try:
            response = await llm.ainvoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            # strip markdown fences if present
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            data = json.loads(text)
            plan = TimeSeriesPlan.model_validate(data)
            logger.info(
                "TS plan: time_col=%s target=%s freq=%s",
                plan.time_column, plan.target_column, plan.frequency,
            )
            return plan
        except Exception as exc:
            logger.warning("TS agent parse failed (%s) — using defaults", exc)
            # Graceful fallback — caller must supply at least a target column
            return TimeSeriesPlan(
                time_column="date",
                target_column="value",
                frequency="daily",
            )

    async def select_model(
        self,
        ts_plan: TimeSeriesPlan,
        ts_analysis_report: dict,
        llm_config: dict | None = None,
    ) -> dict:
        """Choose final model config given stationarity + decomposition results."""
        logger.info("TS agent: selecting forecasting model")

        llm = get_llm(llm_config)

        messages = [
            {"role": "system", "content": TIME_SERIES_MODEL_SELECTION_PROMPT},
            {"role": "user", "content": (
                f"Time series plan:\n{ts_plan.model_dump_json(indent=2)}\n\n"
                f"Analysis results:\n{json.dumps(ts_analysis_report, indent=2)}\n\n"
                f"Output the model selection JSON now."
            )},
        ]

        try:
            response = await llm.ainvoke(messages)
            text = response.content if hasattr(response, "content") else str(response)
            text = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            return json.loads(text)
        except Exception as exc:
            logger.warning("TS model selection parse failed (%s) — using defaults", exc)
            return {
                "primary_model": "xgboost_lags",
                "lag_features": ts_plan.lag_features,
                "rolling_windows": ts_plan.rolling_windows,
                "n_estimators": 200,
                "reasoning": "Default fallback due to LLM parse error.",
            }


time_series_agent = TimeSeriesAgent()
