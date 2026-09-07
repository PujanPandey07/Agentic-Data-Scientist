import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, PolynomialFeatures
from sklearn.feature_selection import VarianceThreshold

from schema.feature_planner import FeatureEngineeringPlan, FeatureEngineeringStep

logger = logging.getLogger(__name__)


class FeatureEngineeringService:
    """Deterministic executor for feature engineering plans.

    Philosophy: Read the plan. Execute exactly. No improvisation.
    """

    def apply_plan(
        self,
        df: pd.DataFrame,
        plan: FeatureEngineeringPlan,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, dict]:
        """Execute a feature engineering plan step-by-step.

        target_column, if given, is a hard guard: no step is allowed to
        touch it, regardless of what the LLM-generated plan says. This is
        defense-in-depth against target leakage (e.g. the planner
        label-encoding the target itself and leaving a near-perfect proxy
        feature behind).
        """
        original_shape = df.shape
        report_steps = []
        all_added = []
        all_removed = []

        # Start fresh — never mutate the input
        current_df = df.copy()

        # Safety check: empty plan
        if not plan or not plan.steps:
            logger.info("Empty feature engineering plan, nothing to execute")
            return current_df, {
                "original_shape": original_shape,
                "final_shape": original_shape,
                "strategy": plan.strategy if plan else "balanced",
                "steps_executed": 0,
                "steps_failed": 0,
                "step_details": [],
                "columns_added": [],
                "columns_removed": [],
                "warnings": plan.warnings if plan else [],
                "notes": plan.notes if plan else [],
                "target_column_guard_triggered": 0,
            }

        logger.info(
            f"Starting feature engineering: {len(plan.steps)} steps planned, "
            f"strategy={plan.strategy}, shape={original_shape}, "
            f"target_column={target_column!r}"
        )

        guard_trigger_count = 0

        # Execute each step in order
        for step in plan.steps:
            safe_step, target_excluded = self._strip_target_column(
                step, target_column
            )
            if target_excluded:
                guard_trigger_count += 1
                logger.warning(
                    f"Step '{step.action}' named target column "
                    f"'{target_column}' in its columns — excluded it before "
                    f"execution to prevent data leakage. Original columns: "
                    f"{step.columns}, sanitized: {safe_step.columns}"
                )

            try:
                result = self._execute_step(current_df, safe_step)
                current_df = result["dataframe"]
                report_steps.append({
                    "action": step.action,
                    "columns": step.columns,
                    "status": "success",
                    "details": result.get("details", ""),
                    "target_column_excluded": target_excluded,
                })
                all_added.extend(result.get("columns_added", []))
                all_removed.extend(result.get("columns_removed", []))
                logger.info(
                    f"Step '{step.action}' succeeded: {result.get('details', '')}")
            except Exception as e:
                # Fault isolation: one step fails, pipeline continues
                report_steps.append({
                    "action": step.action,
                    "columns": step.columns,
                    "status": "failed",
                    "error": str(e),
                    "target_column_excluded": target_excluded,
                })
                logger.warning(f"Step '{step.action}' failed, skipped: {e}")

        final_shape = current_df.shape

        report = {
            "original_shape": original_shape,
            "final_shape": final_shape,
            "strategy": plan.strategy,
            "steps_executed": len([s for s in report_steps if s["status"] == "success"]),
            "steps_failed": len([s for s in report_steps if s["status"] == "failed"]),
            "step_details": report_steps,
            "columns_added": all_added,
            "columns_removed": all_removed,
            "warnings": plan.warnings,
            "notes": plan.notes,
            "target_column_guard_triggered": guard_trigger_count,
        }

        if guard_trigger_count:
            logger.warning(
                f"Target column guard triggered {guard_trigger_count} time(s) "
                f"this run — the planner referenced the target column despite "
                f"being told not to. Worth checking the prompt/EDA context."
            )

        logger.info(
            f"Feature engineering done: {report['steps_executed']} succeeded, "
            f"{report['steps_failed']} failed, final shape={final_shape}"
        )

        return current_df, report

    def _strip_target_column(
        self, step: FeatureEngineeringStep, target_column: str | None
    ) -> tuple[FeatureEngineeringStep, bool]:
        """Return a sanitized copy of the step with target_column removed
        from its columns list, if present. Never mutates the original step.
        """
        if not target_column or target_column not in step.columns:
            return step, False

        safe_columns = [c for c in step.columns if c != target_column]
        return step.model_copy(update={"columns": safe_columns}), True

    def _execute_step(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        """Dispatch to the correct action handler."""
        method_name = f"_action_{step.action}"
        method = getattr(self, method_name, None)
        if method is None:
            raise ValueError(
                f"Unknown feature engineering action: {step.action}")
        return method(df, step)

    # =====================================================================
    # ACTION IMPLEMENTATIONS
    # Each follows: validate -> params -> execute -> track -> return
    # =====================================================================

    def _action_drop_columns(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        cols = [c for c in step.columns if c in df.columns]
        new_df = df.drop(columns=cols) if cols else df.copy()
        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": cols,
            "details": f"Dropped {len(cols)} columns: {cols}" if cols else "No valid columns to drop",
        }

    def _action_one_hot_encode(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        cols = [c for c in step.columns if c in df.columns]
        if not cols:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": "No valid columns found for one-hot encoding, skipped",
            }

        drop_first = (step.params or {}).get("drop_first", False)
        new_df = pd.get_dummies(df, columns=cols, drop_first=drop_first)
        added = [c for c in new_df.columns if c not in df.columns]

        return {
            "dataframe": new_df,
            "columns_added": added,
            "columns_removed": [],
            "details": f"One-hot encoded {cols}, added {len(added)} columns",
        }

    def _action_label_encode(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        new_df = df.copy()
        added = []

        for col in step.columns:
            if col not in new_df.columns:
                continue
            new_col = f"{col}_encoded"
            # Category codes are deterministic and stable for a given dataset
            new_df[new_col] = new_df[col].astype("category").cat.codes
            added.append(new_col)

        return {
            "dataframe": new_df,
            "columns_added": added,
            "columns_removed": [],
            "details": f"Label encoded {step.columns} into {added}" if added else "No valid columns found",
        }

    def _action_standard_scale(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        return self._scale(df, step, StandardScaler(), "Standard")

    def _action_minmax_scale(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        return self._scale(df, step, MinMaxScaler(), "MinMax")

    def _action_robust_scale(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        return self._scale(df, step, RobustScaler(), "Robust")

    def _scale(self, df: pd.DataFrame, step: FeatureEngineeringStep, scaler, name: str):
        cols = [
            c for c in step.columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cols:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": "No numeric columns found to scale, skipped",
            }

        new_df = df.copy()
        scaled_values = scaler.fit_transform(new_df[cols])

        for i, col in enumerate(cols):
            new_df[col] = scaled_values[:, i]

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": f"{name} scaled {cols}",
        }

    def _action_log_transform(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        new_df = df.copy()
        cols = [
            c for c in step.columns
            if c in new_df.columns and pd.api.types.is_numeric_dtype(new_df[c])
        ]
        for col in cols:
            new_df[col] = np.log1p(new_df[col].clip(lower=0))

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": f"Log1p transformed {cols}" if cols else "No numeric columns found",
        }

    def _action_sqrt_transform(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        new_df = df.copy()
        cols = [
            c for c in step.columns
            if c in new_df.columns and pd.api.types.is_numeric_dtype(new_df[c])
        ]
        for col in cols:
            new_df[col] = np.sqrt(new_df[col].clip(lower=0))

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": f"Sqrt transformed {cols}" if cols else "No numeric columns found",
        }

    def _action_power_transform(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        new_df = df.copy()
        power = (step.params or {}).get("power", 2)
        cols = [
            c for c in step.columns
            if c in new_df.columns and pd.api.types.is_numeric_dtype(new_df[c])
        ]
        for col in cols:
            new_df[col] = np.power(new_df[col], power)

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": f"Power transformed {cols} with power={power}" if cols else "No numeric columns found",
        }

    def _action_binning(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        new_df = df.copy()
        bins = (step.params or {}).get("bins", 5)
        strategy = (step.params or {}).get("strategy", "quantile")
        added = []

        for col in step.columns:
            if col not in new_df.columns or not pd.api.types.is_numeric_dtype(new_df[col]):
                continue
            new_col = f"{col}_binned"
            if strategy == "quantile":
                new_df[new_col] = pd.qcut(
                    new_df[col], q=bins, labels=False, duplicates="drop"
                )
            else:
                new_df[new_col] = pd.cut(new_df[col], bins=bins, labels=False)
            added.append(new_col)

        return {
            "dataframe": new_df,
            "columns_added": added,
            "columns_removed": [],
            "details": f"Binned {step.columns} into {bins} bins ({strategy})" if added else "No valid numeric columns",
        }

    def _action_create_interaction(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        new_df = df.copy()
        cols = [c for c in step.columns if c in new_df.columns]
        if len(cols) < 2:
            return {
                "dataframe": new_df,
                "columns_added": [],
                "columns_removed": [],
                "details": "Need at least 2 columns for interaction, skipped",
            }

        operation = (step.params or {}).get("operation", "multiply")
        ops = {
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / (b + 1e-9),
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
        }
        op_func = ops.get(operation, ops["multiply"])

        new_col = f"{'_'.join(cols[:2])}_{operation}"
        new_df[new_col] = op_func(new_df[cols[0]], new_df[cols[1]])

        return {
            "dataframe": new_df,
            "columns_added": [new_col],
            "columns_removed": [],
            "details": f"Created interaction column {new_col}",
        }

    def _action_create_polynomial(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        cols = [
            c for c in step.columns
            if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
        ]
        if not cols:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": "No numeric columns found for polynomial features, skipped",
            }

        degree = (step.params or {}).get("degree", 2)
        include_bias = (step.params or {}).get("include_bias", False)

        poly = PolynomialFeatures(degree=degree, include_bias=include_bias)
        poly_features = poly.fit_transform(df[cols])
        feature_names = poly.get_feature_names_out(cols)

        poly_df = pd.DataFrame(
            poly_features, columns=feature_names, index=df.index)

        poly_df = poly_df.drop(columns=cols, errors="ignore")

        new_df = pd.concat([df.drop(columns=cols), poly_df], axis=1)

        return {
            "dataframe": new_df,
            "columns_added": list(poly_df.columns),
            "columns_removed": cols,
            "details": f"Polynomial degree={degree} for {cols}, generated {len(poly_df.columns)} features",
        }

    def _action_drop_low_variance(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.empty:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": "No numeric columns to check variance, skipped",
            }

        threshold = (step.params or {}).get("threshold", 0.01)
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(numeric_df)

        cols_to_keep = numeric_df.columns[selector.get_support()].tolist()
        cols_to_drop = numeric_df.columns[~selector.get_support()].tolist()

        non_numeric = df.select_dtypes(exclude=[np.number]).columns.tolist()
        new_df = df[non_numeric + cols_to_keep]

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": cols_to_drop,
            "details": f"Dropped low-variance columns: {cols_to_drop}" if cols_to_drop else "No low-variance columns found",
        }

    def _action_drop_high_correlation(self, df: pd.DataFrame, step: FeatureEngineeringStep):
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] < 2:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": "Less than 2 numeric columns, nothing to drop",
            }

        threshold = (step.params or {}).get("threshold", 0.90)
        corr_matrix = numeric_df.corr().abs()

        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        to_drop = [
            column for column in upper.columns
            if any(upper[column] > threshold)
        ]

        if not to_drop:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": f"No correlations above {threshold}, nothing dropped",
            }

        new_df = df.drop(columns=to_drop)

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": to_drop,
            "details": f"Dropped highly correlated columns (|r| > {threshold}): {to_drop}",
        }
