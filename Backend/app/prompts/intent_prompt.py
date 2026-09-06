INTENT_ROUTER_PROMPT = """You classify what a user wants, given their message and whether a prior analysis already exists.

Classify into exactly one of:
- "run_pipeline": user wants to analyze a dataset, build/train a model, or start a new analysis.
- "explain_result": user is asking about something from a PREVIOUS analysis already completed in this session (a metric, a chosen algorithm, a chart) — only valid if a prior report already exists.
- "refine_step": user wants to CHANGE or REDO part of a previous analysis (a different algorithm, an extra chart, different cleaning) — only valid if a prior report already exists.
- "general_question": a question unrelated to any specific dataset (e.g. "what is F1 score?").

If no previous report exists in this session, never choose "explain_result" or "refine_step" — treat it as "general_question" or "run_pipeline" instead.
"""
