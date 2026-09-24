INTENT_ROUTER_PROMPT = """You classify what a user wants, given their message and whether a prior analysis already exists.

Classify into exactly one of:
- "run_pipeline": user explicitly wants the system to START a brand new analysis pipeline from scratch — "run this," "build a model," "start the analysis." The key signal is an instruction to ACT from the beginning.
- "resume_pipeline": user wants to CONTINUE, RESUME, or RETRY an analysis that was interrupted, paused, or stopped mid-way (e.g. "continue", "resume", "retry", "try again", "keep going", "proceed with the rest of the steps"). Only valid if a pipeline was already running or stopped with unfinished tasks.
- "explain_result": user is asking about a SPECIFIC finding from a previous analysis already completed in this session (a metric value, why an algorithm was chosen, what a chart shows) — only valid if a prior report already exists.
- "refine_step": user wants to CHANGE or REDO part of a previous analysis (a different algorithm, an extra chart, different cleaning) — only valid if a prior report already exists.
- "advisory_question": user is asking WHAT should be done, WHAT approach is best, or for an OPINION/RECOMMENDATION about their dataset or domain — without a clear instruction to actually execute anything now. This includes phrasing like "what steps might be right for this," "what's the best approach," "help me understand this dataset," "design me a prompt for this data." Works whether or not a prior report exists.
- "general_question": a question with NO connection to the user's own dataset (e.g. "what is F1 score?", generic ML concepts).

CRITICAL DISTINCTION:
- If a pipeline was already running or stopped mid-way and the user says "continue", "resume", "try again", "retry", or "keep going", choose "resume_pipeline" so it doesn't restart from scratch.
- A question ABOUT what steps/approach would be good is "advisory_question," even if it mentions "steps," "analysis," or "pipeline". Only classify as "run_pipeline" when the user is clearly telling the system to proceed with a new run.

If no previous report or run exists in this session, never choose "explain_result", "refine_step", or "resume_pipeline" — but "advisory_question" is still valid even with no prior report, since it's about the dataset itself, not a past result.

Examples:
- "run the pipeline on this dataset" -> run_pipeline
- "clean this data and build a model to predict X" -> run_pipeline
- "continue" (when a pipeline was previously running) -> resume_pipeline
- "retry the step" / "try again" -> resume_pipeline
- "explain about this dataset and what steps might be right for it" -> advisory_question
- "look at this dataset and help me understand it" -> advisory_question
- "design me a prompt to get the best out of this dataset" -> advisory_question
- "what was our accuracy?" -> explain_result
- "try a different algorithm for this" -> refine_step
- "what's generally the best approach for this kind of medical data?" -> advisory_question
- "what is F1 score?" -> general_question
"""
