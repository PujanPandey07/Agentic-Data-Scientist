# Architecture Decision Records (ADRs)

---

## ADR-001: Service Layer Pattern
**Date**: 2026-07-30  
**Context**: Need to keep FastAPI API routes thin, testable, and decoupled from business logic.  
**Decision**: Business logic lives inside modular service classes (`dataset_service.py`, `cleaning_service.py`, etc.) rather than inside API endpoint functions.

---

## ADR-002: In-Memory Inspection via DataFrames
**Date**: 2026-07-31  
**Context**: Inspecting datasets directly from disk paths creates tight coupling with storage drivers.  
**Decision**: `DatasetInspector` operates strictly on in-memory Pandas DataFrames, maintaining clean separation of concerns.

---

## ADR-003: Separation of Reasoning and Computation
**Date**: 2026-08-05  
**Context**: Prompting an LLM to generate and execute arbitrary Python code is unsafe, non-deterministic, and error-prone.  
**Decision**: LLMs decide *what* needs to be done (producing Pydantic plans), while deterministic Python services execute all data transformations and ML training.

---

## ADR-004: State Checkpointing via PostgreSQL (AsyncPostgresSaver)
**Date**: 2026-09-11  
**Context**: In-memory LangGraph checkpointers lose all conversation state and pipeline progress whenever the server reboots.  
**Decision**: Use `AsyncPostgresSaver` backed by PostgreSQL 16. Graph states are indexed by unique `thread_id`s, allowing pipelines to resume seamlessly across crashes or user sessions.

---

## ADR-005: Decoupled Heavy ML Compute via Redis & ARQ Workers
**Date**: 2026-09-20  
**Context**: Training machine learning models and running Optuna tuning trials blocks the FastAPI event loop and causes HTTP request timeouts.  
**Decision**: Offload model training to a dedicated `arq` background worker pool using Redis as the message broker. The API returns instantly, and workers process jobs asynchronously.

---

## ADR-006: Server-Sent Events (SSE) over HTTP Polling
**Date**: 2026-09-21  
**Context**: Frontend needs real-time training progress updates without spamming the server with polling requests.  
**Decision**: Implement a Server-Sent Events (SSE) streaming endpoint backed by Redis pub/sub. The worker publishes completion events, instantly pushing notifications to the browser.

---

## ADR-007: Three Compiled LangGraph Families Sharing Checkpointer
**Date**: 2026-09-26  
**Context**: Supervised ML, clustering, and time-series forecasting have completely different task requirements, state schemas, and evaluation metrics.  
**Decision**: Build three distinct compiled LangGraph workflows (`supervised_graph`, `unsupervised_graph`, `time_series_graph`) that share the same PostgreSQL checkpointer. The conversation's `pipeline_family` is persisted at creation to route all follow-up turns consistently.

---

## ADR-008: BYOK with Fernet AES-128 Encryption & Live Validation
**Date**: 2026-09-24  
**Context**: Users should be able to supply their own API keys (OpenAI, Anthropic, Gemini, Groq) without exposing secrets in plain text or risking mid-run failures from bad keys.  
**Decision**: Encrypt all user keys at rest using Fernet symmetric encryption with a dedicated master secret. Validate keys live against provider endpoints before persisting.

---

## ADR-009: Reverse Proxy Gateway (Nginx) for Single-Origin Web & API
**Date**: 2026-09-30  
**Context**: Running frontend on port 5173 and backend on port 8000 creates CORS friction, cookie credential issues, and exposes internal database ports.  
**Decision**: Introduce an Nginx container as a single reverse proxy on port 80. Nginx serves the Vite static files at `/` and proxies `/api/` traffic directly to FastAPI, keeping PostgreSQL and Redis on a private internal network.

---

## ADR-010: Streaming Ingestion & Tabular Guardrails
**Date**: 2026-10-01  
**Context**: Unchecked uploads of arbitrary file types or multi-gigabyte datasets can trigger OOM kills on containers.  
**Decision**: Implement 1MB chunked streaming upload with a hard 100MB limit (`HTTP 413`), pre-flight DataFrame probing, dimension bounds (1 to 500k rows; 2 to 1k columns), and automated encoding fallbacks (UTF-8 → Latin-1).
