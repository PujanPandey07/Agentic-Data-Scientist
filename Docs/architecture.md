# AI Data Scientist Architecture

## Vision

The goal of this project is to build an AI Data Scientist capable of autonomously analysing structured datasets through an agentic workflow. Rather than acting as a simple chatbot, the system should reason through the same steps that a human data scientist would follow before presenting conclusions and recommendations.

The architecture is designed around a workflow-first approach. Instead of beginning with implementation details, the project first models the reasoning process and then maps that process to agents, tools, and graph nodes.

---

# Core Principles

- Think before acting.
- Use deterministic code whenever possible.
- Use LLM reasoning only where intelligence is required.
- Keep agents specialized with a single responsibility.
- Design for modularity and extensibility.
- Maintain explainable reasoning throughout the workflow.

---

# High-Level Workflow

````textUser
 │
 ├── Query
 └── Dataset
       │
       ▼
 Dataset Service
       │
       ▼
 Dataset Inspector
       │
       ▼
    GraphState
       │
       ▼
 Planner Agent
       │
       ▼
 AnalysisPlan
       │
       ▼
 Execution Initialization
       │
       ▼
     Router
       │
       ├── Cleaning
       ├── EDA
       ├── Visualization
       ├── Feature Engineering
       ├── Training
       ├── Evaluation
       └── Reporting```

---

# Workflow Explanation

## 1. User Input

The workflow begins when a user uploads a dataset and specifies an objective.

Example objectives:

- Predict customer churn.
- Analyze sales trends.
- Explain important factors affecting house prices.
- Find patterns in employee attrition.

---

## 2. Understand the User's Goal

Before performing any analysis, the system identifies the type of problem.

Possible categories include:

- Classification
- Regression
- Clustering
- General exploratory analysis
- Statistical analysis
- Dataset explanation

This step determines the overall direction of the workflow.

---

## 3. Dataset Inspection

The uploaded dataset is examined to understand its structure.

Typical information includes:

- Number of rows
- Number of columns
- Column names
- Data types
- Missing values
- Duplicate records
- Potential target column
- Class distribution (if applicable)

This step creates a summary that later stages can reuse.

---

## 4. Analysis Planning

Instead of immediately executing tasks, the system first creates an analysis plan.

A typical plan may include:

1. Clean the dataset.
2. Handle missing values.
3. Encode categorical features.
4. Perform exploratory data analysis.
5. Train multiple candidate models.
6. Compare model performance.
7. Generate a final report.

This planning step allows later stages to execute in a structured and explainable manner.

---

## 5. Data Cleaning & Preprocessing

The dataset is prepared for analysis by applying appropriate preprocessing techniques.

Examples include:

- Handling missing values
- Removing duplicate rows
- Correcting inconsistent values
- Encoding categorical variables
- Scaling numerical features
- Detecting outliers

---

## 6. Exploratory Data Analysis (EDA)

The system explores relationships within the dataset by generating descriptive statistics and visualizations.

Examples include:

- Distribution analysis
- Correlation analysis
- Histograms
- Box plots
- Scatter plots
- Heatmaps

The objective is to understand the dataset before training any model.

---

## 7. Feature Engineering (Optional)

Where appropriate, new features may be derived from existing data to improve model performance.

Examples include:

- Date decomposition
- Age groups
- Feature combinations
- Aggregated statistics

---

## 8. Machine Learning

Based on the identified problem type, the system selects suitable machine learning algorithms.

Possible tasks include:

- Model selection
- Dataset splitting
- Training
- Hyperparameter tuning (future enhancement)

---

## 9. Model Evaluation

Candidate models are evaluated using appropriate metrics.

Examples:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC
- Mean Squared Error
- R² Score

The best-performing model is selected based on these results.

---

## 10. Result Interpretation

The trained model is interpreted to explain its behaviour.

Possible outputs include:

- Feature importance
- Prediction explanations
- Model strengths
- Model limitations

The emphasis is on producing understandable insights rather than only reporting numerical scores.

---

## 11. Insights & Recommendations

The system summarizes important findings and generates actionable recommendations.

Examples:

- Key business insights
- Data quality observations
- Important feature relationships
- Suggestions for improving data quality
- Recommendations for future analysis

---

## 12. Report Generation

All outputs from previous stages are combined into a structured report.

The report may contain:

- Dataset overview
- Visualizations
- Statistical summaries
- Model evaluation
- Insights
- Recommendations

---

## 13. Interactive Question Answering

After analysis is complete, users may ask follow-up questions about the generated report, dataset, or model results.

Future versions will support conversation memory and retrieval to provide context-aware responses.

---

# Future Architecture Extensions

The following capabilities are intentionally planned for future milestones:

- Multi-agent collaboration
- LangGraph orchestration
- Persistent state
- Long-term memory
- Retrieval-Augmented Generation (RAG)
- Model Context Protocol (MCP)
- Human-in-the-loop (HITL)
- Streaming responses
- Asynchronous execution
- Reflection and self-improvement
- Multi-dataset analysis

---

**Note:** This document defines the logical workflow of the AI Data Scientist. Implementation details such as graph design, agents, tools, state management, and APIs will be documented separately as the project evolves.
````
