# graphs/plan_validation.py
"""Checks a proposed AnalysisPlan for problems that would break execution
downstream if the user approved it as-is. Each check is independent and
returns a problem string (or None if that check passes) — add new checks
here as new failure modes get discovered, rather than growing a single
tangled if/else in plan_review_node.
"""

import re
MODELING_TASKS = {"model_selection", "hyperparameter_tuning", "evaluation"}


def _check_target_column_present(plan, dataset_summary) -> str | None:
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
    if plan.target_column not in (dataset_summary.column_names or []):
        return (
            f"The detected target column '{plan.target_column}' doesn't "
            f"exist in this dataset. Please specify the correct column name."
        )
    return None


def _check_problem_type_present(plan, dataset_summary) -> str | None:
    needs_type = any(task in (plan.tasks or []) for task in MODELING_TASKS)
    if needs_type and not plan.problem_type:
        return (
            "This request involves modeling, but the problem type "
            "(classification vs regression) couldn't be determined. "
            "Please clarify."
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


# graphs/plan_validation.py — add this function alongside validate_plan


_TARGET_COLUMN_PATTERNS = [
    r"target column is (\w+)",
    r"target column[:\s]+(\w+)",
    r"predict (\w+)",
    r"target is (\w+)",
]


# graphs/plan_validation.py — replace extract_explicit_target_column

def extract_explicit_target_column(text: str, dataset_summary) -> str | None:
    """Deterministically check whether the user's free-text instruction
    explicitly names a real column as the target — rather than relying on
    the general-purpose LLM planner to reliably notice this on every
    revision. Only returns a value that ACTUALLY exists in the dataset,
    so a typo or unrelated word never gets treated as a valid target.
    """
    if not text or not dataset_summary:
        return None

    text_stripped = text.strip().lower()
    columns = dataset_summary.column_names or []
    column_lookup = {c.lower(): c for c in columns}

    # Case 1: the whole instruction IS just the column name — e.g. the
    # user typed "work_mode" with no surrounding phrase at all.
    if text_stripped in column_lookup:
        return column_lookup[text_stripped]

    # Case 2: a phrase naming the column, e.g. "target column is work_mode",
    # "predict career_level".
    for pattern in _TARGET_COLUMN_PATTERNS:
        match = re.search(pattern, text_stripped)
        if match:
            candidate = match.group(1).strip().lower()
            if candidate in column_lookup:
                return column_lookup[candidate]

    return None
