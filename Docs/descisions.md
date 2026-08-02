## ADR-001: Service Layer

Business logic lives inside services instead of API routes to keep endpoints thin and reusable.

---

## ADR-002: Dataset Inspection

Dataset inspection operates on a Pandas DataFrame instead of file paths to maintain separation of responsibilities.

---

## ADR-003: Analysis Orchestration

AnalysisService coordinates analysis modules rather than embedding orchestration logic inside DatasetService.
