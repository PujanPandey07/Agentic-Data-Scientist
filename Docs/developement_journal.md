# Development Journal

---

## 2026-08-02
### Completed
- Integrated dataset inspection pipeline directly into the planner workflow.
- Established clean hand-off of `DatasetSummary` into Pydantic structured output models.

---

## 2026-08-05
### Completed
- Designed dynamic execution workflow using LangGraph.
- Implemented execution queue management using `current_task`, `remaining_tasks`, and `completed_tasks`.
- Created `ExecutionService` to manage task progression and execution logs.
- Implemented the first task node pattern (`cleaning_node`).
- Added router node with conditional task routing.
- Built complete workflow connecting: Dataset → Planner → Execution Initialization → Router → Dynamic Task Nodes.

---

## 2026-08-09
### Completed
- Fully implemented and tested production `CleaningService`.
- Added duplicate removal, missing value detection, and type normalization.
- Verified on Iris dataset (150 rows → 147 rows, 3 duplicates successfully removed).
- Documented architectural update separating state mutation from service operations.

---

## 2026-08-10 – 2026-08-12
### Completed
- **Automated EDA Service**: Implemented statistical summary generation, skewness checks, distribution inspection, and correlation matrices.
- **Deterministic Visualization Service**: Replaced raw LLM code generation with deterministic Matplotlib and Seaborn plotting routines.
- Built automated chart generation for distributions, box plots, scatter matrices, and correlation heatmaps.

---

## 2026-08-28 – 2026-08-31
### Completed
- **Feature Engineering Service**: Built transformation pipelines (imputation, standard scaling, one-hot encoding).
- **Prevented Data Leakage**: Enforced train/test split *prior* to feature engineering fit transformations.
- **Model Selection & Training**: Developed automated candidate selection (Random Forest, Logistic Regression, LightGBM, XGBoost) with 5-fold cross-validation benchmarks.
- **Evaluation Service**: Generated classification/regression metrics (Accuracy, F1, ROC-AUC, RMSE, MAE, R²).
- **Executive Reporting Service**: Integrated automated markdown synthesis combining conclusions, key metrics, and chart artifacts.

---

## 2026-09-01
### Completed
- Integrated **Optuna** for automated Bayesian hyperparameter tuning.
- Added `hyperparameter_tuning_node` into the LangGraph pipeline with trial score tracking.

---

## 2026-09-05 – 2026-09-08
### Completed
- **Intent Router Agent**: Added pre-classification to distinguish between full pipeline execution, advisory Q&A, and general questions, avoiding unnecessary heavy planning calls.
- **Constraint Extractor**: Extracted user-specified algorithm and feature choices from natural language prompts (`constraints.py`).
- **Target Refinement**: Added agentic confirmation when target columns are ambiguous.

---

## 2026-09-09
### Completed
- **Human-In-The-Loop (HITL) Implementation**: Integrated LangGraph `interrupt` pattern before model training.
- Pipeline now pauses mid-execution, persists state, and presents the plan for user approval, adjustment, or rejection.

---

## 2026-09-10 – 2026-09-11
### Completed
- Added persistent long-term and short-term memory services with automated context summarization.
- Configured FastAPI lifespan with `AsyncPostgresSaver` for thread-safe state persistence across sessions.

---

## 2026-09-14 – 2026-09-16
### Completed
- Initialized **React + Vite** single-page application with Tailwind CSS.
- Added JWT authentication (access & refresh tokens) and protected conversation endpoints.
- Implemented context window budgeting to keep LLM prompts under 8,000 tokens.

---

## 2026-09-17 – 2026-09-18
### Completed
- Migrated checkpointer and operational database from SQLite to **PostgreSQL 16**.
- Added **Alembic** database migration framework to manage schema evolutions safely without data loss.

---

## 2026-09-19 – 2026-09-21
### Completed
- Decoupled compute-intensive training from FastAPI: integrated **Redis** and **ARQ** asynchronous background workers.
- Implemented **Server-Sent Events (SSE)** endpoint (`/api/jobs/{job_id}/stream`) with Redis pub/sub for real-time progress streaming to the frontend.

---

## 2026-09-23 – 2026-09-25
### Completed
- **Bring Your Own Key (BYOK)**: Added support for user-provided API keys (OpenAI, Anthropic, Gemini, Groq).
- Secured key storage using **Fernet AES-128 encryption**.
- Implemented live key validation against provider endpoints before persisting credentials.

---

## 2026-09-25 – 2026-09-26
### Completed
- **Unsupervised Learning Pipeline**: Built second compiled LangGraph family for clustering tasks.
- Integrated K-Means, DBSCAN, and Agglomerative clustering with Silhouette and Davies-Bouldin evaluation.

---

## 2026-09-28 – 2026-09-30
### Completed
- **Time Series Forecasting Pipeline**: Implemented third compiled LangGraph family supporting ADF stationarity testing, decomposition, and horizon forecasting.
- **ArtifactPanel UI Redesign**: Added tabbed layout (Summary, Details, Charts) with PDF report export and model artifact downloads.
- **Docker Compose Orchestration**: Containerized the entire 5-service stack (Postgres, Redis, API, Worker, Nginx Frontend) with shared Parquet volumes and healthchecks.
- Auto-run Alembic migrations on startup via container lifespan.

---

## 2026-10-01
### Completed
- **Multi-Format Tabular Ingestion**: Added native support for `.csv`, `.tsv`, `.tab`, `.xlsx`, `.xls`, `.parquet`, `.pq`, `.json`, and `.feather`.
- **Dataset Guardrails**: Implemented 100MB streaming size validation, 0-byte file rejection, dimension bounds (1 to 500,000 rows; 2 to 1,000 cols), and encoding fallbacks.
- **Resilient BYOK Error Surfacing**: Added specific detection for token quota exhaustion, invalid keys, and oversized contexts to prevent silent fallback or confusing crashes.
- **Automated Cache Management**: Scheduled ARQ cron job and startup sweep to purge stale Parquet files older than 24 hours.
- **CI/CD Pipeline**: Configured GitHub Actions workflow for automated backend testing, frontend Vite bundling, and Docker container verification.
- **Production README**: Published full architectural documentation and open-source deployment guide.
