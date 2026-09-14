# utilis/prompt_context.py
"""Shared helpers for building LLM-prompt-safe summaries out of large
EDA/dataset objects. GraphState itself has no size limit — this exists
because LLM API calls DO have a token limit, and a wide or
high-cardinality dataset's raw eda_report/dataset_summary can easily
blow past it if dumped into a prompt as-is.
"""

MAX_VALUE_COUNTS_PER_COLUMN = 10
MAX_CATEGORICAL_COLUMNS_DETAILED = 20
MAX_NUMERICAL_COLUMNS_DETAILED = 30


def summarize_eda_report_for_prompt(eda_report: dict) -> dict:
    """Cap an eda_report's size before it's inserted into an LLM prompt.
    Keeps the shape recognizable to the LLM but bounds the worst-case
    blow-up sources: per-category value_counts on high-cardinality
    columns, and huge column counts.
    """
    if not eda_report:
        return eda_report

    summary = {
        "shape": eda_report.get("shape"),
        "numerical_columns": eda_report.get("numerical_columns", []),
        "categorical_columns": eda_report.get("categorical_columns", []),
        "missing_values": eda_report.get("missing_values"),
    }

    # Numerical summary: usually small per-column (a handful of stats),
    # but cap the NUMBER of columns shown in detail anyway for very wide
    # datasets.
    numerical_summary = eda_report.get("numerical_summary") or {}
    capped_numerical = dict(
        list(numerical_summary.items())[:MAX_NUMERICAL_COLUMNS_DETAILED]
    )
    summary["numerical_summary"] = capped_numerical
    if len(numerical_summary) > MAX_NUMERICAL_COLUMNS_DETAILED:
        summary["numerical_summary_truncated"] = (
            f"{len(numerical_summary) - MAX_NUMERICAL_COLUMNS_DETAILED} "
            "more numerical columns omitted for brevity"
        )

    # Categorical summary: THIS is the actual blow-up risk — a
    # high-cardinality column (free text, unique IDs, etc.) can have
    # hundreds/thousands of distinct value_counts entries. Cap both the
    # number of columns shown AND the number of values per column.
    categorical_summary = eda_report.get("categorical_summary") or {}
    capped_categorical = {}
    for i, (col, info) in enumerate(categorical_summary.items()):
        if i >= MAX_CATEGORICAL_COLUMNS_DETAILED:
            break
        value_counts = info.get("value_counts", {})
        top_values = dict(list(value_counts.items())[
                          :MAX_VALUE_COUNTS_PER_COLUMN])
        capped_categorical[col] = {
            "unique_values": info.get("unique_values"),
            "top_value_counts": top_values,
            "value_counts_truncated": len(value_counts) > MAX_VALUE_COUNTS_PER_COLUMN,
        }
    summary["categorical_summary"] = capped_categorical
    if len(categorical_summary) > MAX_CATEGORICAL_COLUMNS_DETAILED:
        summary["categorical_summary_truncated"] = (
            f"{len(categorical_summary) - MAX_CATEGORICAL_COLUMNS_DETAILED} "
            "more categorical columns omitted for brevity"
        )

    # Correlation matrix scales O(n^2) with numeric columns — genuinely
    # not needed by most planners (feature engineering, visualization),
    # so it's dropped entirely here rather than capped. Callers that
    # DO need it (if any) should pull it separately, not via this helper.
    return summary
