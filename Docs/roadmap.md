# Project Roadmap & Milestones

---

### Core Pipeline Milestones
- **Phase 1 ─ System Foundation** ✅
  - FastAPI asynchronous server, project layout, environment configuration.
- **Phase 2 ─ Dataset Ingestion & Understanding** ✅
  - Tabular parsing, metadata inspection, structural validation, and caching.
- **Phase 3 ─ Planning Agent** ✅
  - Structured Pydantic `AnalysisPlan`, LLM reasoning with repair retries.
- **Phase 4 ─ Workflow Orchestration** ✅
  - Dynamic LangGraph state graph, conditional task routing, and execution queue.
- **Phase 5 ─ Data Cleaning Service** ✅
  - Deterministic duplicate removal, missing value imputation, type alignment.
- **Phase 6 ─ Exploratory Data Analysis (EDA)** ✅
  - Summary statistics, skewness detection, correlation matrix generation.
- **Phase 7 ─ Visualization Suite** ✅
  - Deterministic Matplotlib & Seaborn generation for distributions, boxplots, and heatmaps.
- **Phase 8 ─ Feature Engineering & Machine Learning** ✅
  - Leak-free train/test splitting, automated candidate model training with cross-validation.
- **Phase 9 ─ Hyperparameter Tuning & Evaluation** ✅
  - Optuna Bayesian optimization, multi-metric scoring (Accuracy, F1, ROC-AUC, RMSE, MAE).
- **Phase 10 ─ Executive Reporting & Artifact Generation** ✅
  - Markdown report synthesis, PDF export, trained model downloads.

---

### Advanced Agentic & System Milestones
- **Phase 11 ─ Human-In-The-Loop (HITL) Guardrails** ✅
  - LangGraph state interruption, plan review, parameter editing, and user approval.
- **Phase 12 ─ Multi-Family Pipelines** ✅
  - Supervised learning graph, unsupervised clustering graph, and time-series forecasting graph.
- **Phase 13 ─ Distributed Background Compute** ✅
  - Redis broker, ARQ async worker pool, and real-time Server-Sent Events (SSE).
- **Phase 14 ─ Security & BYOK Architecture** ✅
  - Fernet AES encryption, live provider validation, and quota exhaustion detection.
- **Phase 15 ─ Tabular Guardrails & Format Expansion** ✅
  - Support for CSV, TSV, Excel, Parquet, JSON, Feather; 100MB streaming limits and bounds checking.
- **Phase 16 ─ Production Containerization & CI/CD** ✅
  - 5-service Docker Compose stack, Nginx reverse proxy gateway, automated Alembic migrations, GitHub Actions CI.
- **Phase 17 ─ Cloud Deployment & Public Hosting** 🚀 *(In Progress)*
  - Deploy to free/low-cost VM (Oracle Cloud / GCP / Hetzner), automated continuous deployment (CD).
