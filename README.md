# Agentic-Data-Scientist
> An AI-powered data analysis platform that uses agentic workflows to clean, analyse, visualize, and explain datasets.

## Overview

AI Data Scientist is an end-to-end agentic AI application designed to emulate the workflow of a junior data scientist. Instead of simply answering questions about a dataset, the system plans analyses, performs exploratory data analysis (EDA), generates visualizations, selects appropriate statistical methods, trains machine learning models, and produces comprehensive reports.

The primary goal of this project is to explore modern LLM engineering concepts while building a production-style AI application from the ground up.

---

## Project Goals

This project aims to:

* Build a modular agentic AI system using LangGraph.
* Automate the complete data analysis workflow.
* Combine deterministic Python pipelines with LLM reasoning.
* Learn production-grade LLM application architecture.
* Explore modern AI engineering concepts through a real-world project.

---

## Planned Features

### Data Analysis

* Dataset upload (CSV, Excel, JSON)
* Automatic dataset profiling
* Data cleaning suggestions
* Exploratory Data Analysis (EDA)
* Statistical analysis
* Interactive visualizations
* Machine Learning model selection
* Model evaluation and comparison
* Automated report generation

### AI Capabilities

* Agentic workflows using LangGraph
* Multi-agent architecture
* Planning and reasoning
* Tool calling
* Structured outputs with Pydantic
* Reflection and self-improvement
* Human-in-the-loop (HITL)
* Long-term memory
* Retrieval-Augmented Generation (RAG)
* Model Context Protocol (MCP) integration
* Streaming responses
* Asynchronous execution

---

## Planned Architecture

```text
                        User

                          │

                    Coordinator Agent

                          │

 ┌────────────────────────────────────────────────────┐
 │                                                    │
 │  Dataset Agent                                     │
 │  Data Cleaning Agent                               │
 │  EDA Agent                                         │
 │  Visualization Agent                               │
 │  Statistics Agent                                  │
 │  Machine Learning Agent                            │
 │  Insight Agent                                     │
 │  Report Generation Agent                           │
 │                                                    │
 └────────────────────────────────────────────────────┘

                          │

                    Final Report
```

---

## Technology Stack

### Backend

* Python
* FastAPI
* LangGraph
* LangChain

### AI

* Google Gemini
* Groq

### Data Science

* Pandas
* NumPy
* Scikit-learn
* Matplotlib
* Plotly

### Frontend

* React
* Tailwind CSS

### Database & Storage

* PostgreSQL
* FAISS / Vector Database

---

## Development Roadmap

### Phase 1 — Foundation

* [ ] Project setup
* [ ] FastAPI backend
* [ ] React frontend
* [ ] File upload system

### Phase 2 — Dataset Understanding

* [ ] Dataset profiling
* [ ] Metadata extraction
* [ ] Automatic column detection

### Phase 3 — Planning Agent

* [ ] Task planning
* [ ] Workflow generation
* [ ] Agent routing

### Phase 4 — Data Analysis

* [ ] Data cleaning
* [ ] Exploratory Data Analysis
* [ ] Visualization generation
* [ ] Statistical analysis

### Phase 5 — Machine Learning

* [ ] Automatic model selection
* [ ] Model training
* [ ] Model evaluation
* [ ] Feature importance

### Phase 6 — Advanced AI Features

* [ ] Reflection
* [ ] Memory
* [ ] RAG
* [ ] MCP
* [ ] HITL
* [ ] Streaming
* [ ] Async execution

### Phase 7 — Production

* [ ] Authentication
* [ ] Persistent state
* [ ] Docker deployment
* [ ] Documentation

---

## Repository Structure

```text
AI-Data-Scientist/

├── backend/
├── frontend/
├── docs/
├── README.md
└── docker-compose.yml
```

---

## Learning Objectives

This project is being built as a practical exploration of modern AI engineering concepts, including:

* Agentic AI
* LangGraph
* State Management
* Multi-Agent Systems
* Tool Calling
* Pydantic
* Streaming
* Async Programming
* Human-in-the-loop (HITL)
* Retrieval-Augmented Generation (RAG)
* Model Context Protocol (MCP)
* Production AI Architecture

---

## Current Status

🚧 **Project initialization in progress**

Current milestone:

* Repository setup
* Initial architecture
* Development roadmap

---

## Contributing

This project is currently under active development and is primarily intended as a learning and portfolio project. Suggestions, feedback, and discussions are welcome.

---

## License

This project is licensed under the MIT License.

