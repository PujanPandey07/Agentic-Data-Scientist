# graphs/plan_validation.py
"""Checks a proposed AnalysisPlan for problems that would break execution
downstream if the user approved it as-is. Each check is independent and
returns a problem string (or None if that check passes) — add new checks
here as new failure modes get discovered, rather than growing a single
tangled if/else in plan_review_node.
"""

import re

# Tasks that require a target_column + a supervised problem_type
# (classification/regression) to make sense. Clustering plans can
# legitimately include "evaluation" (silhouette score, etc.) and even a
# "model_selection"-style stage without ever having a target column — so
# these checks are explicitly scoped to non-clustering plans.
MODELING_TASKS = {"model_selection", "hyperparameter_tuning", "evaluation"}


def _normalize(name: str) -> str:
    """Normalize a column name for comparison: lowercase, spaces→underscore."""
    return name.strip().lower().replace(" ", "_")


def _check_target_column_present(plan, dataset_summary) -> str | None:
    if plan.problem_type == "clustering":
        return None
    needs_target = any(task in (plan.tasks or []) for task in MODELING_TASKS)
    if needs_target and not plan.target_column:
        return (
            "This request requires a target column to train/evaluate a "
            "model, but none could be detected. Please specify which "
            "column to predict."
        )
    return None


def _check_target_column_exists(plan, dataset_summary) -> str | None:
    if not plan.target_column or not dataset_summary:
        return None
    columns = dataset_summary.column_names or []
    # Build a normalized lookup: snake_case form -> original name
    normalized_lookup = {_normalize(c): c for c in columns}
    target_normalized = _normalize(plan.target_column)
    # Accept if exact match OR normalized match (handles original-cased vs
    # snake_case difference between pre- and post-cleaning summaries)
    if plan.target_column in columns or target_normalized in normalized_lookup:
        return None
    return (
        f"The detected target column '{plan.target_column}' doesn't "
        f"exist in this dataset. Please specify the correct column name."
    )


def _check_problem_type_present(plan, dataset_summary) -> str | None:
    # Clustering plans, and any other non-modeling plan, don't need this
    # check — problem_type="clustering" already satisfies "a type was
    # determined" on its own.
    needs_type = any(task in (plan.tasks or []) for task in MODELING_TASKS)
    if needs_type and not plan.problem_type:
        return (
            "This request involves modeling, but the problem type "
            "(classification, regression, or clustering) couldn't be "
            "determined. Please clarify."
        )
    return None


def _check_forced_algorithm_supported(plan, dataset_summary) -> str | None:
    # Placeholder for the "constraints name an unsupported algorithm"
    # case flagged earlier — left as a stub until detect_forced_algorithm's
    # known-algorithm list is confirmed, so this file already has the slot
    # ready rather than being retrofitted later.
    return None


# Every check runs on every plan review. Order doesn't matter — all
# problems found are surfaced together, not just the first one.
CHECKS = [
    _check_target_column_present,
    _check_target_column_exists,
    _check_problem_type_present,
    _check_forced_algorithm_supported,
]


def validate_plan(plan, dataset_summary) -> list[str]:
    """Returns a list of human-readable problems with the plan. Empty
    list means the plan is runnable as-is."""
    if not plan:
        return []
    problems = []
    for check in CHECKS:
        problem = check(plan, dataset_summary)
        if problem:
            problems.append(problem)
    return problems


_TARGET_COLUMN_PATTERNS = [
    r"target column is ['\"]?([\w.]+)['\"]?",
    r"target column[:\s]+['\"]?([\w.]+)['\"]?",
    r"predict (?:column )?['\"]?([\w.]+)['\"]?",
    r"target is ['\"]?([\w.]+)['\"]?",
    r"(?:using|with) column ['\"]?([\w.]+)['\"]? as (?:the )?target",
    r"column ['\"]?([\w.]+)['\"]? as (?:the )?target",
    r"as target column ['\"]?([\w.]+)['\"]?",
    r"target[:\s=]+['\"]?([\w.]+)['\"]?",
]


def extract_explicit_target_column(text: str, dataset_summary) -> str | None:
    """Deterministically check whether the user's free-text instruction
    explicitly names a real column as the target — rather than relying on
    the general-purpose LLM planner to reliably notice this on every
    revision. Only returns a value that ACTUALLY exists in the dataset,
    so a typo or unrelated word never gets treated as a valid target.

    Matching is done against both original column names and their
    normalized (snake_case) forms, and covers digits (e.g. '1', '64')
    and dotted column names (e.g. '1.1').
    """
    if not text or not dataset_summary:
        return None

    text_stripped = text.strip().lower()
    # Strip optional surrounding quotes e.g. "1" or '1'
    if (text_stripped.startswith('"') and text_stripped.endswith('"')) or \
       (text_stripped.startswith("'") and text_stripped.endswith("'")):
        text_stripped = text_stripped[1:-1].strip()

    columns = dataset_summary.column_names or []
    # Build both an exact lookup and a normalized lookup
    column_lookup = {str(c).lower(): str(c) for c in columns}
    normalized_lookup = {_normalize(str(c)): str(c) for c in columns}

    # Case 1: the whole instruction IS just the column name — e.g. the
    # user typed "work_mode" or "1" with no surrounding phrase at all.
    if text_stripped in column_lookup:
        return column_lookup[text_stripped]
    if _normalize(text_stripped) in normalized_lookup:
        return normalized_lookup[_normalize(text_stripped)]

    # Case 2: a phrase naming the column, e.g. "target column is 1",
    # "predict daily minimum temperatures", "with column 1 as target".
    for pattern in _TARGET_COLUMN_PATTERNS:
        match = re.search(pattern, text_stripped)
        if match:
            raw = match.group(1).strip()
            for candidate in _candidate_substrings(raw):
                if candidate in column_lookup:
                    return column_lookup[candidate]
                norm = _normalize(candidate)
                if norm in normalized_lookup:
                    return normalized_lookup[norm]

    # Case 3: scan if any column name is mentioned in isolation as a whole token/word
    # e.g. "column 1" or "use 1 as target"
    for col_str, orig in column_lookup.items():
        # Match col_str bordered by non-word/boundary or spaces
        pattern = r"(?:^|\b|\s|['\"])" + re.escape(col_str) + r"(?:$|\b|\s|['\"])"
        if re.search(pattern, text_stripped):
            # Check context implies target
            if any(k in text_stripped for k in ["target", "predict", "outcome", "label", "dependent"]):
                return orig

    return None


def _candidate_substrings(text: str) -> list[str]:
    """Yield the full text, then progressively drop trailing words.
    This handles cases where the regex captured trailing words that are
    not part of the column name (e.g. "sales amount next year" → tries
    "sales amount next year", "sales amount next", "sales amount", "sales").
    """
    words = text.split()
    candidates = []
    for length in range(len(words), 0, -1):
        candidates.append(" ".join(words[:length]))
    return candidates

