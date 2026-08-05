# Agentic Data Scientist

> An AI-powered data analysis platform that uses LangGraph-based agentic workflows to inspect, clean, analyze, visualize, model, and explain datasets.

---

# Overview

Agentic Data Scientist is an end-to-end AI application designed to emulate the workflow of a junior data scientist.

Instead of simply answering questions about a dataset, the system builds an execution plan, dynamically executes data analysis tasks, trains machine learning models, evaluates results, and generates comprehensive reports.

The project is also a hands-on exploration of modern LLM engineering concepts, including agentic workflows, LangGraph, structured outputs, state management, and production-ready AI application architecture.

---

# Project Goals

- Build a modular AI-powered data analysis platform.
- Learn production-grade AI application architecture.
- Combine deterministic Python pipelines with LLM reasoning.
- Build reusable services for data science workflows.
- Explore LangGraph through a real-world project.

---

# Current Architecture

```text
                           User
                             │
                             ▼
                     FastAPI Backend
                             │
                             ▼
                    Dataset Upload API
                             │
                             ▼
                     Dataset Inspector
                             │
                             ▼
                      Planner Agent (LLM)
                             │
                    Generates Analysis Plan
                             │
                             ▼
                  LangGraph Execution Engine
                             │
                             ▼
                     Execution Queue
                             │
                             ▼
                      Dynamic Router
                             │
        ┌─────────────────────────────────────────┐
        │                                         │
        │ Cleaning Node                           │
        │ EDA Node                                │
        │ Visualization Node                      │
        │ Feature Engineering Node                │
        │ Training Node                           │
        │ Evaluation Node                         │
        │ Reporting Node                          │
        │                                         │
        └─────────────────────────────────────────┘
                             │
                             ▼
                    Python Service Layer
                             │
        ┌─────────────────────────────────────────┐
        │ Dataset Service                         │
        │ Cleaning Service                        │
        │ EDA Service                             │
        │ Visualization Service                   │
        │ Model Training Service                  │
        │ Evaluation Service                      │
        └─────────────────────────────────────────┘
                             │
                             ▼
                        Final Report
```

---

# Current Progress

## Completed

### Backend

- FastAPI project setup
- Modular project structure
- Dataset upload endpoint
- CSV and Excel loading
- Dataset inspection
- Dataset metadata extraction

### LangGraph

- Graph state design
- Planner agent
- Structured LLM outputs using Pydantic
- Execution queue
- Dynamic task routing
- Conditional workflow execution
- Reusable execution service
- Workflow orchestration

### AI

- Planner Agent
- Structured planning outputs
- Task generation based on user intent and dataset summary

---

# Planned Features

## Data Processing

- Data cleaning
- Missing value handling
- Duplicate removal
- Data type correction

## Exploratory Data Analysis

- Descriptive statistics
- Correlation analysis
- Distribution analysis
- Dataset insights

## Visualization

- Histograms
- Box plots
- Correlation heatmaps
- Scatter plots
- Target analysis

## Machine Learning

- Feature engineering
- Model selection
- Model training
- Hyperparameter tuning
- Model evaluation
- Explainability

## Reporting

- Automated reports
- AI-generated insights
- Recommendations
- Exportable summaries

---

# Technology Stack

## Backend

- Python
- FastAPI
- LangGraph
- LangChain

## AI

- Google Gemini
- Groq

## Data Science

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly

## Frontend

- React
- Tailwind CSS

## Storage

- PostgreSQL
- FAISS (planned)

---

# Development Roadmap

## Phase 1 — Foundation ✅

- Project setup
- FastAPI backend
- Dataset upload
- Dataset loading
- Dataset inspection

## Phase 2 — Workflow Engine ✅

- Graph state
- Planner agent
- Execution queue
- Dynamic router
- LangGraph workflow

## Phase 3 — Data Processing (In Progress)

- Data cleaning
- EDA
- Visualization
- Statistical analysis

## Phase 4 — Machine Learning

- Feature engineering
- Model selection
- Model training
- Evaluation

## Phase 5 — AI Enhancements

- Insight generation
- Report generation
- Reflection
- Human-in-the-loop
- Memory
- RAG
- MCP
- Streaming

## Phase 6 — Production

- Authentication
- Persistent state
- Docker deployment
- Testing
- Documentation

---

# Repository Structure

```text
Agentic-Data-Scientist/

├── backend/
│   ├── app/
│   ├── uploads/
│   └── requirements.txt
│
├── frontend/
│
├── Docs/
│
├── docker-compose.yml
│
└── README.md
```

---

# Learning Objectives

This project serves as a practical exploration of:

- Agentic AI
- LangGraph
- State Management
- Workflow Orchestration
- LLM Planning
- Structured Outputs
- Pydantic
- FastAPI
- Async Programming
- Retrieval-Augmented Generation (RAG)
- Human-in-the-loop (HITL)
- Production AI Architecture

---

# Current Status

🚧 **Active Development**

### Current Milestone

Building the data processing pipeline.

Completed:

- Dataset upload
- Dataset inspection
- Planner agent
- Dynamic workflow engine

Next:

- Cleaning pipeline
- EDA pipeline
- Visualization pipeline

---

# License

This project is licensed under the MIT License.
