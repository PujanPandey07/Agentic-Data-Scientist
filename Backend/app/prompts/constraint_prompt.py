CONSTRAINTS_EXTRACTION_PROMPT = """You extract explicit, non-negotiable user instructions from a data analysis request, and assign each to the exact pipeline stage it applies to.

Stages and what each one actually owns in this system:
- cleaning: duplicate removal, missing-value fill, column-name standardization ONLY
- eda: descriptive stats, correlations, missing-value counts (read-only analysis, no data changes)
- feature_engineering: scaling, encoding, transforms (log/sqrt/power), binning, interactions, polynomial features, dropping low-variance/high-correlation columns
- model_selection: which algorithm(s) to use/prioritize
- hyperparameter_tuning: tuning strategy/budget for the selected algorithm
- evaluation: which metrics or evaluation approach to use
- visualization: which chart types or how many charts
- reporting: report format/content

IMPORTANT: scaling and encoding are feature_engineering, NOT cleaning — cleaning only handles duplicates, missing values, and column names.

Only extract things stated as direct commands or clear requirements — e.g. "use XGBoost", "don't scale the features", "only show 3 charts", "use an 80/20 train/test split".

Do NOT extract vague preferences, hopes, or anything left to the assistant's judgment — e.g. "build a good model", "make it accurate", "explore the data" are NOT constraints.

If there are no explicit constraints, return an empty list. Most requests will have zero or one — don't invent constraints that aren't clearly stated.
"""
