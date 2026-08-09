# Agentic Data Scientist

> An AI-powered data analysis platform that uses agentic workflows to understand datasets, plan analysis, execute data-processing tasks, and generate data-driven insights.

## Overview

**Agentic Data Scientist** is an AI-powered data analysis platform designed to emulate the workflow of a data scientist.

Instead of simply sending a dataset to an LLM and asking it to analyze the data, the system uses a **LangGraph-based agentic workflow** to separate reasoning, orchestration, and deterministic data processing.

The system accepts a user's natural-language request together with a dataset, understands the dataset structure, creates an analysis plan, and executes the required tasks through a stateful workflow.

The project is also being developed as a practical exploration of modern AI engineering concepts such as:

- Agentic workflows
- LangGraph
- Structured LLM outputs
- Pydantic
- State management
- Tool calling
- Async execution
- RAG
- MCP
- Human-in-the-loop systems
- Persistent memory

---

# Core Architecture

The current architecture follows a simple principle:

> **LLMs reason. Python executes. LangGraph orchestrates.**

The LLM should decide **what needs to be done**, while deterministic Python services perform operations such as cleaning, analysis, and machine learning.

```text
                         User
                           │
                           │
                    User Query + Dataset
                           │
                           ▼
                  ┌──────────────────┐
                  │   Dataset Node   │
                  └────────┬─────────┘
                           │
                           ▼
                  Dataset Inspection
                           │
                           ▼
                  ┌──────────────────┐
                  │   Planner Agent  │
                  │                  │
                  │  LLM + Pydantic  │
                  └────────┬─────────┘
                           │
                           ▼
                    Analysis Plan
                           │
                           ▼
                Initialize Execution
                           │
                           ▼
                     Task Router
                           │
          ┌────────────────┼────────────────┐
          ▼                ▼                ▼
      Cleaning           EDA         Visualization
          │                │                │
          └────────────────┼────────────────┘
                           │
                           ▼
                  Feature Engineering
                           │
                           ▼
                    Model Selection
                           │
                           ▼
                       Training
                           │
                           ▼
                     Evaluation
                           │
                           ▼
                      Reporting
```

The workflow is state-driven, meaning each node reads from and updates a shared `GraphState`.

---

# Current Workflow

The implemented workflow currently follows this sequence:

```text
START
  ↓
Dataset
  ↓
Planner
  ↓
Initialize Execution
  ↓
Router
  ↓
Task Node
  ↓
Execution Service
  ↓
Router
  ↓
Next Task
  ↓
...
  ↓
END
```

The planner determines which tasks are required.

The router then selects the appropriate node based on:

```text
current_task
```

After a task finishes, the execution service:

1. Marks the task as completed.
2. Removes the next task from the execution queue.
3. Updates `current_task`.
4. Adds an execution log.
5. Returns control to the router.

---

# Graph State

The workflow uses a shared `GraphState` to maintain context throughout execution.

Current state structure:

```text
GraphState
│
├── User Input
│   ├── user_query
│   └── dataset_id
│
├── Dataset
│   ├── dataframe
│   └── dataset_summary
│
├── Planning
│   └── analysis_plan
│
├── Cleaning
│   └── cleaning_report
│
└── Execution
    ├── current_task
    ├── remaining_tasks
    ├── completed_tasks
    └── execution_logs
```

The dataset summary is stored in the graph state so that downstream nodes have access to the same dataset context without repeatedly inspecting the dataset or relying on incomplete assumptions.

---

# Dataset Pipeline

The dataset pipeline is already implemented for CSV and Excel datasets.

```text
Upload
  ↓
Dataset ID
  ↓
File Storage
  ↓
Dataset Loading
  ↓
Pandas DataFrame
  ↓
Dataset Inspector
  ↓
DatasetSummary
```

The dataset service handles:

- File validation
- Dataset ID generation
- File storage
- Dataset retrieval
- CSV loading
- Excel loading

Supported formats currently include:

```text
.csv
.xlsx
.xls
```

---

# Dataset Summary

The dataset inspector generates a structured `DatasetSummary`.

It currently captures information such as:

- Number of rows
- Number of columns
- Column names
- Data types
- Numerical columns
- Categorical columns
- Missing values
- Duplicate rows
- Memory usage
- Missing-value presence
- Duplicate presence
- Potential target columns

Example:

```python
DatasetSummary(
    rows=150,
    columns=5,
    column_names=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width",
        "species"
    ],
    numerical_columns=[
        "sepal_length",
        "sepal_width",
        "petal_length",
        "petal_width"
    ],
    categorical_columns=["species"],
    duplicate_rows=3,
    has_missing_values=False,
    has_duplicates=True
)
```

---

# Planner Agent

The planner is responsible for converting the user's natural-language request and dataset summary into a structured execution plan.

The LLM does not directly manipulate the dataset.

Instead, it produces an `AnalysisPlan` using Pydantic.

Conceptually:

```text
User Query
     +
Dataset Summary
     │
     ▼
 Planner LLM
     │
     ▼
AnalysisPlan
```

The plan contains:

```text
user_intent
problem_type
target_column
tasks
reasoning
```

For example:

```python
AnalysisPlan(
    user_intent="Predict whether a customer will purchase a product.",
    problem_type="classification",
    target_column="Purchased",
    tasks=[
        "cleaning",
        "eda",
        "visualization",
        "feature_engineering",
        "model_selection",
        "training",
        "evaluation",
        "reporting"
    ],
    reasoning="..."
)
```

The planner therefore acts as the **reasoning layer**, while LangGraph controls execution.

---

# Cleaning Pipeline

The first real execution stage has now been implemented.

The cleaning architecture follows:

```text
Cleaning Node
      ↓
CleaningService
      ↓
Cleaned DataFrame + Cleaning Report
      ↓
GraphState
```

The `CleaningService` is responsible for deterministic data-cleaning operations.

Current functionality includes:

- Duplicate removal
- Missing-value handling
- Data-type validation
- Invalid-entry detection
- Summary generation

The first implemented workflow operation is duplicate removal.

For example:

```text
150 rows
   ↓
3 duplicate rows removed
   ↓
147 rows
```

The cleaning stage also generates a structured report:

```python
{
    "original_rows": 150,
    "final_rows": 147,
    "duplicates_removed": 3,
    "missing_values_remaining": 0
}
```

The report is stored in:

```python
state["cleaning_report"]
```

---

# Execution Service

Execution state management is separated from individual task implementations.

The `ExecutionService` is responsible for advancing the workflow after a task finishes.

It handles:

```text
completed_tasks
current_task
remaining_tasks
execution_logs
```

This allows individual task nodes to remain focused on their own work.

For example:

```text
Cleaning Node
      │
      ├── Perform cleaning
      ├── Store cleaning result
      │
      ▼
Execution Service
      │
      ├── Mark cleaning complete
      ├── Select next task
      └── Log execution
```

This separation is intended to keep the architecture scalable as more execution stages are added.

---

# Current Implementation Status

## Implemented

- [x] Project structure
- [x] FastAPI backend foundation
- [x] Dataset upload service
- [x] Dataset storage
- [x] CSV loading
- [x] Excel loading
- [x] Dataset inspection
- [x] Structured dataset summary
- [x] Pydantic schemas
- [x] LangGraph state
- [x] Planner agent
- [x] Structured planner output
- [x] Analysis plan
- [x] Execution queue
- [x] Task router
- [x] Execution service
- [x] Cleaning node
- [x] Cleaning service
- [x] Duplicate removal
- [x] Cleaning report
- [x] State persistence across workflow nodes
- [x] End-to-end graph testing

## Currently Developing

- [ ] EDA service
- [ ] EDA node
- [ ] Statistical analysis
- [ ] Visualization generation
- [ ] Feature engineering
- [ ] Model selection
- [ ] Model training
- [ ] Model evaluation
- [ ] Automated reporting

---

# Planned Advanced Capabilities

After the core data-science workflow is stable, the project will progressively introduce more advanced AI engineering capabilities.

### Agentic AI

- [ ] Specialized analysis agents
- [ ] Agent coordination
- [ ] Reflection
- [ ] Self-correction
- [ ] Planning validation
- [ ] Retry and recovery mechanisms

### LLM Engineering

- [ ] Structured outputs
- [ ] Tool calling
- [ ] Model routing
- [ ] Streaming
- [ ] Human-in-the-loop
- [ ] Prompt management

### Knowledge & Memory

- [ ] RAG
- [ ] Long-term memory
- [ ] Persistent workflow state
- [ ] Vector storage
- [ ] MCP integration

### Production

- [ ] Authentication
- [ ] Persistent database state
- [ ] Background execution
- [ ] Docker deployment
- [ ] Frontend integration
- [ ] Monitoring
- [ ] Error handling
- [ ] Automated testing

---

# Technology Stack

## Backend

- Python
- FastAPI
- LangGraph
- LangChain
- Pydantic

## Data Science

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Plotly

## AI Models

- Google Gemini
- Groq

## Storage

- PostgreSQL
- FAISS / Vector Database

## Frontend

- React
- Tailwind CSS

---

# Repository Structure

```text
AI-Data-Scientist/
│
├── Backend/
│   ├── app/
│   │   ├── agents/
│   │   │   └── planner.py
│   │   │
│   │   ├── analysis/
│   │   │   └── inspector.py
│   │   │
│   │   ├── graphs/
│   │   │   ├── state.py
│   │   │   ├── nodes.py
│   │   │   └── workflow.py
│   │   │
│   │   ├── schema/
│   │   │   ├── analysis_plan.py
│   │   │   ├── dataset_summary.py
│   │   │   └── upload.py
│   │   │
│   │   ├── services/
│   │   │   ├── dataset_service.py
│   │   │   ├── cleaning_service.py
│   │   │   └── execution_service.py
│   │   │
│   │   ├── uploads/
│   │   └── test_graph.py
│   │
│   └── requirements.txt
│
├── Frontend/
│
├── Docs/
│
├── docker-compose.yml
└── README.md
```

---

# Development Philosophy

The project is intentionally being developed incrementally.

Rather than attempting to build the complete AI Data Scientist in one step, each major capability is first implemented independently and then connected to the LangGraph workflow.

The current development strategy is:

```text
Dataset Infrastructure
        ↓
Dataset Understanding
        ↓
Planner
        ↓
Workflow Orchestration
        ↓
Cleaning
        ↓
EDA
        ↓
Visualization
        ↓
Feature Engineering
        ↓
Machine Learning
        ↓
Reporting
        ↓
Advanced Agentic Features
```

This allows each layer to be tested before additional complexity is introduced.

---

# Learning Objectives

This project serves as a practical exploration of modern AI engineering.

Key concepts being explored include:

- Agentic AI
- LangGraph
- Graph-based workflows
- State management
- Pydantic
- Structured LLM outputs
- Deterministic tool execution
- Task routing
- Async programming
- Tool calling
- Multi-agent systems
- Reflection
- Human-in-the-loop systems
- RAG
- Long-term memory
- MCP
- Production AI architecture

---

# Current Milestone

🚧 **Core agentic workflow implementation in progress**

The project has moved beyond initial setup.

The current working pipeline is:

```text
Dataset Upload
      ↓
Dataset Loading
      ↓
Dataset Inspection
      ↓
Dataset Summary
      ↓
Planner Agent
      ↓
Structured Analysis Plan
      ↓
Execution Queue
      ↓
Task Router
      ↓
Cleaning Node
      ↓
Cleaning Service
      ↓
Updated Graph State
```

The next milestone is to implement the **EDA service and EDA node** and integrate them into the existing execution workflow.

---

# Project Goal

The long-term goal is to build a reliable AI Data Scientist that can take:

```text
Dataset + Natural Language Request
```

and produce:

```text
Understanding
     ↓
Analysis Plan
     ↓
Data Processing
     ↓
Exploration
     ↓
Visualization
     ↓
Modeling
     ↓
Evaluation
     ↓
Insights
     ↓
Final Report
```

while keeping deterministic computation separate from LLM reasoning and maintaining execution context through a stateful agentic workflow.

---

# License

This project is licensed under the MIT License.
