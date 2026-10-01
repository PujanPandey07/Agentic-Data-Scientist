# 🤖 Agentic AI Data Scientist

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![CI/CD Pipeline](https://github.com/PujanPandey07/Agentic-Data-Scientist/actions/workflows/ci.yml/badge.svg)](https://github.com/PujanPandey07/Agentic-Data-Scientist/actions/workflows/ci.yml)
[![Python: 3.14+](https://img.shields.io/badge/Python-3.14%2B-blue?logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688?logo=fastapi)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/Orchestration-LangGraph-orange)](https://langchain-ai.github.io/langgraph/)
[![Docker](https://img.shields.io/badge/Docker-Compose-2496ED?logo=docker)](https://docker.com)
[![React](https://img.shields.io/badge/Frontend-React%20%2B%20Vite-61DAFB?logo=react)](https://react.dev)

> **An autonomous, full-lifecycle Data Science platform powered by multi-agent LangGraph workflows, distributed background compute, and human-in-the-loop control.**

Upload any tabular dataset (`.csv`, `.tsv`, `.parquet`, `.xlsx`, `.json`, `.feather`), state your objective in plain English, and the system autonomously plans, cleans, performs exploratory analysis, engineers features, trains optimized ML models, tunes hyperparameters, evaluates performance, and synthesizes interactive executive reports.

---

## ⚡ Highlights & Differentiators

Unlike naive LLM wrappers that simply generate and run unconstrained scripts, this system follows an architectural tenet: **LLMs reason and plan; deterministic Python services execute; LangGraph orchestrates state.**

- **Three Specialized Agentic Graph Families**: Automatically infers and routes to the right compiled LangGraph workflow:
  - 🎯 **Supervised Learning**: Classification and regression with cross-validation and benchmark comparisons.
  - 🔍 **Unsupervised Learning**: Customer segmentation and clustering using K-Means, DBSCAN, and hierarchical clustering with silhouette/Davies-Bouldin metrics.
  - 📈 **Time Series Forecasting**: Automated stationary testing, decomposition, and horizon forecasting.
- **Human-In-The-Loop (HITL) Guardrails**: LangGraph pauses execution before training, presenting a structured task breakdown, target column, and methodology for user review, modification, or one-click approval.
- **Distributed Async ML Compute**: Decouples API requests from heavy model training using **Redis** and **ARQ background workers**. Real-time job status is streamed directly to the browser via **Server-Sent Events (SSE)**.
- **Bring Your Own Key (BYOK) Security**: Secure user API key storage with **Fernet symmetric encryption**. Keys are validated live against provider endpoints with automatic context, rate-limit, and quota error surfacing.
- **Production Container Stack**: Orchestrated with Docker Compose, including health checks, shared persistent cache volumes, and an **Nginx** reverse proxy serving the React SPA and proxying API traffic.
- **Multi-Format Tabular Ingestion & Safety**: Upload `.csv`, `.tsv`, `.xlsx`, `.parquet`, `.json`, or `.feather` up to 100MB with streaming chunk validation, encoding fallbacks, and sanity checks.

---

## 🏛️ System Architecture

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

## 🔄 The Autonomous Pipeline Lifecycle

```text
[ Dataset Upload ] ──▶ [ Pre-flight Inspection & Guardrails ]
                                │
                                ▼
                       [ Intent Router Agent ]
                                │
       ┌────────────────────────┼────────────────────────┐
       ▼                        ▼                        ▼
 (General Q&A)            (Direct Advisory)       (Run Pipeline)
       │                        │                        │
       └──────────────┬─────────┘                        ▼
                      ▼                        [ Planner Agent (LLM) ]
             [ Direct Response ]                         │
                                                         ▼
                                               [ HITL Approval Pause ]
                                                         │
                                               (User Approves / Edits)
                                                         │
                                                         ▼
                                            ┌─────────────────────────┐
                                            │ 1. Data Cleaning        │
                                            │ 2. Automated EDA        │
                                            │ 3. Visualization Suite  │
                                            │ 4. Feature Engineering  │
                                            │ 5. Model Selection (ARQ)│
                                            │ 6. Hyperparameter Tuning│
                                            │ 7. Evaluation & Metrics │
                                            │ 8. Executive Reporting  │
                                            └─────────────────────────┘
```

---

## 🚀 Quickstart (Production Docker Compose)

The entire system runs locally or on any cloud VPS with one command:

### 1. Clone the repository
```bash
git clone https://github.com/PujanPandey07/Agentic-Data-Scientist.git
cd Agentic-Data-Scientist
```

### 2. Configure Environment Variables
Copy the template files and insert your API keys:
```bash
# Root Compose credentials
cp .env.example .env

# Backend secrets & model provider
cp Backend/.env.example Backend/.env
```

Ensure your `Backend/.env` contains your preferred default model keys:
```ini
LLM_PROVIDER=groq                     # Options: groq, gemini, openai, anthropic
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=llama-3.3-70b-versatile
API_KEY_ENCRYPTION_SECRET=your_32_byte_base64_fernet_secret
JWT_SECRET=your_super_secret_jwt_key
```

### 3. Build & Run
```bash
docker compose up --build -d
```

Open your browser to:
- **Frontend App**: `http://localhost`
- **Backend API Docs**: `http://localhost:8000/docs`

Database tables and Alembic schema migrations apply automatically on startup.

---

## 💻 Local Development Setup

If you wish to develop without Docker:

### Backend
```bash
cd Backend
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt

# Start Redis & Postgres (e.g., via Docker)
docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=devpassword123 -e POSTGRES_DB=ai_data_scientist postgres:16-alpine
docker run -d -p 6379:6379 redis:7-alpine

# Run Migrations
alembic upgrade head

# Start API Server
uvicorn app.main:app --reload --port 8000

# Start Background Worker (in a separate terminal)
cd app
arq worker.WorkerSettings
```

### Frontend
```bash
cd Frontend
npm install
npm run dev
```
Open `http://localhost:5173`.

---

## 📊 Supported File Formats & Guardrails

| Format | Extension | Engine | Encoding Support |
|---|---|---|---|
| **CSV** | `.csv` | `pd.read_csv` | UTF-8 with automatic fallback to Latin-1/CP1252 |
| **TSV / Tab** | `.tsv`, `.tab` | `pd.read_csv(sep='\t')` | Tab-delimited with encoding fallback |
| **Excel** | `.xlsx`, `.xls` | `openpyxl` / `xlrd` | Multi-sheet workbook support |
| **Parquet** | `.parquet`, `.pq` | `pyarrow` | High-performance columnar data |
| **JSON** | `.json` | `pd.read_json` | Standard record arrays and JSON Lines (`ndjson`) |
| **Feather** | `.feather` | `pyarrow` | Ultra-fast memory-mapped binary |

### Built-in Guardrails:
- **Stream-checked File Size**: 100MB upload limit with streaming early termination (`HTTP 413`) to prevent memory exhaustion.
- **Empty File Protection**: 0-byte uploads are immediately rejected (`HTTP 400`).
- **Structural Integrity Probe**: Files are validated as tabular DataFrames upon upload before conversation records are created.
- **Dimension Bounds**: Enforces minimums (≥ 1 row, ≥ 2 columns) and maximums (≤ 1,000 columns, ≤ 500,000 in-memory rows) for stable execution.

---

## 🛠️ Technology Stack

- **Orchestration & Agents**: [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), [Pydantic v2](https://github.com/pydantic/pydantic)
- **Backend API**: [FastAPI](https://fastapi.tiangolo.com), [Uvicorn](https://www.uvicorn.org), [Starlette](https://www.starlette.io)
- **Data Science & ML**: [Pandas](https://pandas.pydata.org), [NumPy](https://numpy.org), [Scikit-Learn](https://scikit-learn.org), [XGBoost](https://xgboost.readthedocs.io), [Optuna](https://optuna.org), [PyArrow](https://arrow.apache.org)
- **Visualization**: [Matplotlib](https://matplotlib.org), [Seaborn](https://seaborn.pydata.org), [Plotly](https://plotly.com)
- **Task Queue & Cache**: [Redis](https://redis.io), [ARQ](https://github.com/samuelcolvin/arq)
- **Database & Checkpointing**: [PostgreSQL 16](https://www.postgresql.org), [SQLAlchemy 2.0 (Async)](https://www.sqlalchemy.org), [Alembic](https://alembic.sqlalchemy.org), [Asyncpg](https://github.com/MagicStack/asyncpg)
- **Frontend**: [React 18](https://react.dev), [Vite](https://vitejs.dev), [Tailwind CSS](https://tailwindcss.com), [Axios](https://axios-http.com), [Lucide Icons](https://lucide.dev)
- **Infrastructure**: [Docker](https://www.docker.com), [Docker Compose](https://docs.docker.com/compose/), [Nginx](https://nginx.org)

---

## 🔒 Security & Privacy

- **Encrypted Secrets**: Sensitive API keys supplied via BYOK are encrypted with AES-128 via Fernet and decrypted only during execution.
- **Strict Network Isolation**: Databases and Redis brokers are inaccessible to public networks in Docker Compose.
- **Zero Raw LLM Code Execution**: The system utilizes deterministic Python modules for all data transformation and modeling rather than passing arbitrary `exec()` or `eval()` code strings from LLMs.

---

## 📄 License

Distributed under the MIT License. See [LICENSE](LICENSE) for more information.
