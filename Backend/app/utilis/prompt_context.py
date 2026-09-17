# utilis/prompt_context.py
"""Shared helpers for building LLM-prompt-safe summaries out of large
EDA/dataset objects. GraphState itself has no size limit — this exists
because LLM API calls DO have a token limit, and a wide or
high-cardinality dataset's raw eda_report/dataset_summary can easily
blow past it if dumped into a prompt as-is.
"""

MAX_VALUE_COUNTS_PER_COLUMN = 5
MAX_CATEGORICAL_COLUMNS_DETAILED = 10
MAX_NUMERICAL_COLUMNS_DETAILED = 15
MAX_COLUMNS_LISTED = 20


def truncate_for_prompt(obj, max_list_items: int = 15, max_dict_keys: int = 25):
    """Truncate an object to a size that is safe for inclusion in an LLM prompt."""
    if hasattr(obj, "model_dump"):
        obj = obj.model_dump()
    """Generic, schema-agnostic size cap for ANY object about to be
    inserted into an LLM prompt — unlike summarize_eda_report_for_prompt
    (which knows eda_report's exact shape), this works blind: it just
    bounds how many list items / dict keys are shown at each level,
    recursively. Use this for objects whose internal structure isn't
    known/stable here (dataset_summary, feature_engineering_report),
    so a wide dataset or a long feature-engineering plan can't blow
    past the token limit no matter what shape it takes.
    """
    if isinstance(obj, dict):
        items = list(obj.items())[:max_dict_keys]
        result = {k: truncate_for_prompt(
            v, max_list_items, max_dict_keys) for k, v in items}
        if len(obj) > max_dict_keys:
            result["_truncated"] = f"{len(obj) - max_dict_keys} more keys omitted for brevity"
        return result
    if isinstance(obj, list):
        truncated = [truncate_for_prompt(
            v, max_list_items, max_dict_keys) for v in obj[:max_list_items]]
        if len(obj) > max_list_items:
            truncated.append(
                f"...{len(obj) - max_list_items} more items omitted for brevity")
        return truncated
    return obj


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
        "numerical_columns": eda_report.get("numerical_columns", [])[:MAX_COLUMNS_LISTED],
        "categorical_columns": eda_report.get("categorical_columns", [])[:MAX_COLUMNS_LISTED],
        "missing_values": dict(list((eda_report.get("missing_values") or {}).items())[:MAX_COLUMNS_LISTED]),
    }

    if len(eda_report.get("numerical_columns", [])) > MAX_COLUMNS_LISTED:
        summary["numerical_columns_truncated"] = True
    if len(eda_report.get("categorical_columns", [])) > MAX_COLUMNS_LISTED:
        summary["categorical_columns_truncated"] = True
    if len(eda_report.get("missing_values") or {}) > MAX_COLUMNS_LISTED:
        summary["missing_values_truncated"] = True

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
