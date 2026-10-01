# Future Roadmap & Architectural Ideas

This document outlines potential future enhancements, architectural evolutions, and advanced capabilities for the **Agentic Data Scientist** platform.

---

## 1. Model Explainability & Interpretability (XAI)
- **Automated SHAP & LIME Integration**: Generate automated SHAP (SHapley Additive exPlanations) summary, waterfall, and force plots for the winning model.
- **Narrative Explainability**: Use LLM reasoning to translate mathematical feature importances and interaction terms into clear business language (e.g., *"A 10% increase in customer tenure reduced churn risk by 18%"*).
- **Partial Dependence Plots (PDP)**: Visualize non-linear relationships between top features and target predictions.

---

## 2. One-Click Model Serving & Export
- **Instant REST Prediction Endpoints**: Automatically generate a hosted `/api/models/{model_id}/predict` endpoint once a model finishes training.
- **ONNX & Edge Serialization**: Export trained scikit-learn, XGBoost, and LightGBM models to open standard formats (ONNX) for sub-millisecond production inference.
- **Stand-Alone Inference Dockerfiles**: Allow users to download a self-contained microservice container pre-packaged with their trained model artifact and a lightweight FastAPI scoring wrapper.

---

## 3. Remote Data Connectors & Enterprise Ingestion
- **SQL & Cloud Warehouse Ingestion**: Allow users to connect directly to PostgreSQL, MySQL, Snowflake, BigQuery, and ClickHouse via secure connection strings or IAM credentials.
- **Automated SQL Pre-Filtering**: The Planner Agent generates read-only SQL queries to filter and aggregate large tables before pulling data into the local pipeline.
- **Streaming S3/R2 Object Storage**: Transition `runtime_cache` and uploads from local persistent volumes to S3-compatible cloud storage (AWS S3, Cloudflare R2, MinIO), enabling horizontal auto-scaling of worker nodes.

---

## 4. Agentic Reflection & Automated Remediation
- **Performance-Driven Reflection Loops**: If cross-validation benchmarks fall below an acceptable threshold (e.g., F1 < 0.65), a reflection agent analyzes confusion matrices and triggers targeted remediation:
  - Automated class-imbalance correction (SMOTE, Class-Weight rebalancing).
  - Outlier isolation via Isolation Forests.
  - Automated non-linear feature interactions and polynomial features.
- **Data Quality Alerts**: Warn users when high multicollinearity (VIF > 10) or target leakage features are detected.

---

## 5. Multi-Modal Tabular + Text Features
- **Zero-Code Text Embeddings**: Automatically detect unstructured text columns (e.g., customer reviews, product descriptions) and pass them through local lightweight embedding models (`all-MiniLM-L6-v2`).
- **Hybrid Feature Unions**: Concatenate dense text embeddings with standard scaled tabular features to train unified multi-modal gradient-boosted models.

---

## 6. Interactive "What-If" Analysis UI
- **Scenario Simulator**: Render an interactive slider panel in the React frontend allowing users to tweak feature values and inspect live model predictions in real time.
- **Counterfactual Explanations**: Provide users with minimum required feature shifts to flip a prediction (e.g., *"What would this customer's credit score need to be to get approved?"*).

---

## 7. Model Monitoring & Data Drift
- **Population Stability Index (PSI)**: Automatically compare distributions between historical training data and newly uploaded inference batches.
- **Automated Retraining Triggers**: Notify users or trigger background ARQ retraining jobs when concept or covariate drift exceeds tolerance limits.
