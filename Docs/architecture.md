# System Architecture

## Vision

The goal of this project is to build an autonomous, reliable AI Data Scientist that analyzes structured tabular datasets through an agentic workflow. Rather than acting as a simple code-generating chatbot, the system reasons through the same structured methodology that a human data scientist would follow, separating planning, orchestration, and heavy computation into distinct architectural layers.

---

## Core Engineering Principles

1. **LLMs Reason, Python Executes, LangGraph Orchestrates**: LLMs determine *what* to do; deterministic Python modules execute all operations; LangGraph maintains state.
2. **Decouple Heavy Compute from Web Serving**: Training algorithms and running Optuna tuning trials must not block the web server event loop.
3. **Deterministic Human-In-The-Loop**: Always pause for human review before training models or mutating datasets.
4. **Resilient State Persistence**: Conversational state and graph checkpoints must survive crashes and reboots via ACID-compliant storage.
5. **Defense in Depth**: User API keys are encrypted at rest; public traffic is gated behind a single-origin reverse proxy; datasets are bounded by streaming guardrails.

---

## High-Level Topology

```text
                                  ┌──────────────────────────┐
                                  │   Browser / React SPA    │
                                  └─────────────┬────────────┘
                                                │ (Port 80)
                                                ▼
                                  ┌──────────────────────────┐
                                  │      Nginx Gateway       │
                                  └──────┬────────────┬──────┘
                                         │            │
                         /api/jobs/stream│ (SSE)      │ /api/*
                                         ▼            ▼
                                  ┌──────────────────────────┐
                                  │    FastAPI Application   │
                                  └──────┬────────────┬──────┘
                                         │            │
                       State Checkpoints │            │ Background Tasks
                                         ▼            ▼
    ┌──────────────────────────────────────────┐    ┌──────────────────────────────────┐
    │          PostgreSQL Checkpointer         │    │          Redis 7 Engine          │
    │  (AsyncPostgresSaver + Persistent State) │    │   (Job Broker + Pub/Sub Events)  │
    └──────────────────────────────────────────┘    └────────────────┬─────────────────┘
                                                                     │
                                                                     ▼
                                                    ┌──────────────────────────────────┐
                                                    │         ARQ Worker Pool          │
                                                    │  (Model Training, Optuna Tuning, │
                                                    │   Clustering & Shared Parquet)   │
                                                    └──────────────────────────────────┘
```

---

## Three Compiled LangGraph Pipeline Families

To prevent a monolithic, unmaintainable state machine, the system maintains three distinct compiled LangGraph workflows that share a unified PostgreSQL checkpointer:

### 1. Supervised Learning Graph
- **Intent**: Classification & Regression tasks.
- **Workflow**: `cleaning` → `eda` → `visualization` → `feature_engineering` → `model_selection` (ARQ) → `hyperparameter_tuning` (ARQ) → `evaluation` → `reporting`.
- **Key Characteristics**: Enforces leak-free train/test splits prior to feature engineering; tests multiple candidate algorithms (Random Forest, LightGBM, XGBoost) with 5-fold cross-validation.

### 2. Unsupervised Clustering Graph
- **Intent**: Customer segmentation, grouping, anomaly discovery.
- **Workflow**: `cleaning` → `eda` → `visualization` → `clustering_model_selection` (ARQ) → `clustering_tuning` (ARQ) → `clustering_evaluation` → `reporting`.
- **Key Characteristics**: Fits on the full dataset without train/test splits; optimizes cluster counts using Silhouette and Davies-Bouldin metrics; generates PCA scatter plots.

### 3. Time Series Forecasting Graph
- **Intent**: Sequential horizon forecasting.
- **Workflow**: `cleaning` → `ts_analysis` (ADF stationarity test, seasonal decomposition) → `ts_model_selection` → `ts_training` → `ts_evaluation` → `reporting`.
- **Key Characteristics**: Time-aware validation without random shuffling; evaluates horizon performance via MAPE and RMSE.

---

## Component Deep Dive

### 1. Ingestion & Pre-flight Guardrails (`dataset_service.py`)
- Reads `.csv`, `.tsv`, `.tab`, `.xlsx`, `.xls`, `.parquet`, `.pq`, `.json`, and `.feather`.
- Streams incoming files in 1MB chunks with early termination at 100MB (`HTTP 413`).
- Validates tabular integrity, rejects 0-byte files, and enforces dimension limits (1 to 500k rows, 2 to 1k columns).

### 2. Intent Routing & Constraint Extraction
- **Intent Router**: Distinguishes between brand new pipeline runs, continuation/resumption of paused jobs, advisory questions, and general Q&A.
- **Constraint Extractor**: Parses natural language constraints from prompts (e.g. *"use XGBoost"*, *"skip visualization"*) and injects them into state before planning.

### 3. Human-In-The-Loop (HITL) Checkpoints
- Prior to triggering compute jobs, LangGraph invokes an `interrupt()` call.
- The pipeline pauses execution and yields an interrupt payload to the API.
- The user reviews the plan in the UI and sends an `approved`, `rejected`, or `edit_instruction` decision, which resumes the graph cleanly via LangGraph `Command(resume=...)`.

### 4. Asynchronous Task Queue & Streaming
- Model training and hyperparameter tuning jobs are enqueued to **Redis**.
- An **ARQ worker** executes jobs out-of-process, reading shared Parquet data from the shared Docker cache volume.
- Upon job completion, the worker publishes an event to Redis pub/sub.
- The FastAPI SSE endpoint (`/api/jobs/{job_id}/stream`) picks up the pub/sub event and pushes status updates directly to the browser.

### 5. Security & BYOK Architecture
- Supports user-supplied keys for OpenAI, Anthropic, Gemini, and Groq.
- Encrypts keys at rest using **Fernet AES-128**.
- Performs pre-flight model validation against live provider APIs before persisting keys.
- Surfaces specific errors for token quota exhaustion, invalid keys, and context overflows.
