import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    PolynomialFeatures,
)
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
        touch it, regardless of what the LLM-generated plan says.
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
                    f"Step '{step.action}' succeeded: "
                    f"{result.get('details', '')}"
                )

            except Exception as e:
                # Fault isolation: one step fails, pipeline continues
                report_steps.append({
                    "action": step.action,
                    "columns": step.columns,
                    "status": "failed",
                    "error": str(e),
                    "target_column_excluded": target_excluded,
                })

                logger.warning(
                    f"Step '{step.action}' failed, skipped: {e}"
                )

        final_shape = current_df.shape

        report = {
            "original_shape": original_shape,
            "final_shape": final_shape,
            "strategy": plan.strategy,
            "steps_executed": len(
                [s for s in report_steps if s["status"] == "success"]
            ),
            "steps_failed": len(
                [s for s in report_steps if s["status"] == "failed"]
            ),
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
            f"Feature engineering done: "
            f"{report['steps_executed']} succeeded, "
            f"{report['steps_failed']} failed, "
            f"final shape={final_shape}"
        )

        return current_df, report

    # =====================================================================
    # TRAIN / TEST SAFE FEATURE ENGINEERING
    # =====================================================================

    def apply_plan_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        plan: FeatureEngineeringPlan,
        target_column: str | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
        """Apply feature engineering using train-fitted transformations.

        Transformations that learn from the data are fitted ONLY on the
        training dataframe and then applied to the test dataframe.

        This prevents feature-engineering leakage.
        """

        original_train_shape = train_df.shape
        original_test_shape = test_df.shape

        train_current = train_df.copy()
        test_current = test_df.copy()

        report_steps = []
        all_added = []
        all_removed = []
        guard_trigger_count = 0

        # Safety check: empty plan
        if not plan or not plan.steps:
            logger.info(
                "Empty feature engineering plan, nothing to execute"
            )

            return train_current, test_current, {
                "original_train_shape": original_train_shape,
                "original_test_shape": original_test_shape,
                "final_train_shape": original_train_shape,
                "final_test_shape": original_test_shape,
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
            f"Starting train/test feature engineering: "
            f"{len(plan.steps)} steps planned, "
            f"train_shape={original_train_shape}, "
            f"test_shape={original_test_shape}"
        )

        # Execute each step in order
        for step in plan.steps:
            safe_step, target_excluded = self._strip_target_column(
                step, target_column
            )

            if target_excluded:
                guard_trigger_count += 1

                logger.warning(
                    f"Step '{step.action}' named target column "
                    f"'{target_column}' — excluded before execution."
                )

            try:
                (
                    train_current,
                    test_current,
                    result,
                ) = self._execute_step_train_test(
                    train_current,
                    test_current,
                    safe_step,
                )

                report_steps.append({
                    "action": step.action,
                    "columns": step.columns,
                    "status": "success",
                    "details": result.get("details", ""),
                    "target_column_excluded": target_excluded,
                })

                all_added.extend(
                    result.get("columns_added", [])
                )

                all_removed.extend(
                    result.get("columns_removed", [])
                )

                logger.info(
                    f"Train/test step '{step.action}' succeeded: "
                    f"{result.get('details', '')}"
                )

            except Exception as e:
                # Fault isolation
                report_steps.append({
                    "action": step.action,
                    "columns": step.columns,
                    "status": "failed",
                    "error": str(e),
                    "target_column_excluded": target_excluded,
                })

                logger.warning(
                    f"Train/test step '{step.action}' failed, skipped: {e}"
                )

        report = {
            "original_train_shape": original_train_shape,
            "original_test_shape": original_test_shape,
            "final_train_shape": train_current.shape,
            "final_test_shape": test_current.shape,
            "strategy": plan.strategy,
            "steps_executed": len(
                [s for s in report_steps if s["status"] == "success"]
            ),
            "steps_failed": len(
                [s for s in report_steps if s["status"] == "failed"]
            ),
            "step_details": report_steps,
            "columns_added": all_added,
            "columns_removed": all_removed,
            "warnings": plan.warnings,
            "notes": plan.notes,
            "target_column_guard_triggered": guard_trigger_count,
        }

        logger.info(
            f"Train/test feature engineering complete: "
            f"{report['steps_executed']} succeeded, "
            f"{report['steps_failed']} failed, "
            f"train_shape={train_current.shape}, "
            f"test_shape={test_current.shape}"
        )

        return train_current, test_current, report

    def _execute_step_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Execute one feature engineering step safely across train/test."""

        action = step.action

        # These transformations learn parameters from the data.
        if action in {
            "standard_scale",
            "minmax_scale",
            "robust_scale",
        }:
            return self._scale_train_test(
                train_df,
                test_df,
                step,
            )

        if action == "one_hot_encode":
            return self._one_hot_train_test(
                train_df,
                test_df,
                step,
            )

        if action == "label_encode":
            return self._label_encode_train_test(
                train_df,
                test_df,
                step,
            )

        if action == "binning":
            return self._binning_train_test(
                train_df,
                test_df,
                step,
            )

        if action == "create_polynomial":
            return self._polynomial_train_test(
                train_df,
                test_df,
                step,
            )

        if action == "drop_low_variance":
            return self._variance_train_test(
                train_df,
                test_df,
                step,
            )

        if action == "drop_high_correlation":
            return self._correlation_train_test(
                train_df,
                test_df,
                step,
            )

        # Stateless transformations can safely be executed separately.
        train_result = self._execute_step(
            train_df,
            step,
        )

        test_result = self._execute_step(
            test_df,
            step,
        )

        return (
            train_result["dataframe"],
            test_result["dataframe"],
            {
                "details": train_result.get("details", ""),
                "columns_added": train_result.get(
                    "columns_added",
                    [],
                ),
                "columns_removed": train_result.get(
                    "columns_removed",
                    [],
                ),
            },
        )

    def _scale_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Fit scaler on train and transform both train and test."""

        scaler_map = {
            "standard_scale": (
                StandardScaler(),
                "Standard",
            ),
            "minmax_scale": (
                MinMaxScaler(),
                "MinMax",
            ),
            "robust_scale": (
                RobustScaler(),
                "Robust",
            ),
        }

        scaler, name = scaler_map[step.action]

        cols = [
            c
            for c in step.columns
            if (
                c in train_df.columns
                and c in test_df.columns
                and pd.api.types.is_numeric_dtype(train_df[c])
                and pd.api.types.is_numeric_dtype(test_df[c])
            )
        ]

        if not cols:
            return (
                train_df.copy(),
                test_df.copy(),
                {
                    "details": (
                        "No numeric columns found to scale, skipped"
                    ),
                    "columns_added": [],
                    "columns_removed": [],
                },
            )

        scaler.fit(train_df[cols])

        train_new = train_df.copy()
        test_new = test_df.copy()

        train_new[cols] = scaler.transform(
            train_df[cols]
        )

        test_new[cols] = scaler.transform(
            test_df[cols]
        )

        return (
            train_new,
            test_new,
            {
                "details": (
                    f"{name} scaled {cols} using train data only"
                ),
                "columns_added": [],
                "columns_removed": [],
            },
        )

    def _one_hot_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Learn one-hot categories from train and align test columns."""

        cols = [
            c
            for c in step.columns
            if c in train_df.columns
            and c in test_df.columns
        ]

        if not cols:
            return (
                train_df.copy(),
                test_df.copy(),
                {
                    "details": (
                        "No valid columns found for "
                        "one-hot encoding"
                    ),
                    "columns_added": [],
                    "columns_removed": [],
                },
            )

        drop_first = (step.params or {}).get(
            "drop_first",
            False,
        )

        train_new = pd.get_dummies(
            train_df,
            columns=cols,
            drop_first=drop_first,
        )

        test_new = pd.get_dummies(
            test_df,
            columns=cols,
            drop_first=drop_first,
        )

        # Test must have exactly the same feature columns as train.
        test_new = test_new.reindex(
            columns=train_new.columns,
            fill_value=0,
        )

        added = [
            c
            for c in train_new.columns
            if c not in train_df.columns
        ]

        return (
            train_new,
            test_new,
            {
                "details": (
                    f"One-hot encoded {cols} using "
                    f"train categories only"
                ),
                "columns_added": added,
                "columns_removed": [],
            },
        )

    def _label_encode_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Learn label mapping from train and apply it to test."""

        train_new = train_df.copy()
        test_new = test_df.copy()
        added = []

        for col in step.columns:
            if (
                col not in train_new.columns
                or col not in test_new.columns
            ):
                continue

            new_col = f"{col}_encoded"

            categories = pd.Series(
                train_new[col].dropna().unique()
            ).sort_values()

            mapping = {
                value: index
                for index, value in enumerate(categories)
            }

            train_new[new_col] = (
                train_new[col]
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

            test_new[new_col] = (
                test_new[col]
                .map(mapping)
                .fillna(-1)
                .astype(int)
            )

            added.append(new_col)

        return (
            train_new,
            test_new,
            {
                "details": (
                    f"Label encoded {step.columns} "
                    f"using train mapping only"
                ),
                "columns_added": added,
                "columns_removed": [],
            },
        )

    def _binning_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Learn bin edges from train and apply them to test."""

        train_new = train_df.copy()
        test_new = test_df.copy()

        bins = (step.params or {}).get(
            "bins",
            5,
        )

        strategy = (step.params or {}).get(
            "strategy",
            "quantile",
        )

        added = []

        for col in step.columns:
            if (
                col not in train_new.columns
                or col not in test_new.columns
                or not pd.api.types.is_numeric_dtype(
                    train_new[col]
                )
                or not pd.api.types.is_numeric_dtype(
                    test_new[col]
                )
            ):
                continue

            new_col = f"{col}_binned"

            if strategy == "quantile":
                _, edges = pd.qcut(
                    train_new[col],
                    q=bins,
                    retbins=True,
                    duplicates="drop",
                )

                # Allow test values outside train range.
                edges[0] = -np.inf
                edges[-1] = np.inf

            else:
                edges = np.linspace(
                    train_new[col].min(),
                    train_new[col].max(),
                    bins + 1,
                )

                edges[0] = -np.inf
                edges[-1] = np.inf

            train_new[new_col] = pd.cut(
                train_new[col],
                bins=edges,
                labels=False,
                include_lowest=True,
            )

            test_new[new_col] = pd.cut(
                test_new[col],
                bins=edges,
                labels=False,
                include_lowest=True,
            )

            added.append(new_col)

        return (
            train_new,
            test_new,
            {
                "details": (
                    f"Binned {step.columns} using "
                    f"train-derived bin edges"
                ),
                "columns_added": added,
                "columns_removed": [],
            },
        )

    def _polynomial_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Fit PolynomialFeatures on train and transform test."""

        cols = [
            c
            for c in step.columns
            if (
                c in train_df.columns
                and c in test_df.columns
                and pd.api.types.is_numeric_dtype(
                    train_df[c]
                )
                and pd.api.types.is_numeric_dtype(
                    test_df[c]
                )
            )
        ]

        if not cols:
            return (
                train_df.copy(),
                test_df.copy(),
                {
                    "details": (
                        "No numeric columns found for "
                        "polynomial features, skipped"
                    ),
                    "columns_added": [],
                    "columns_removed": [],
                },
            )

        degree = (step.params or {}).get(
            "degree",
            2,
        )

        include_bias = (step.params or {}).get(
            "include_bias",
            False,
        )

        poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
        )

        train_values = poly.fit_transform(
            train_df[cols]
        )

        test_values = poly.transform(
            test_df[cols]
        )

        feature_names = poly.get_feature_names_out(
            cols
        )

        train_poly = pd.DataFrame(
            train_values,
            columns=feature_names,
            index=train_df.index,
        )

        test_poly = pd.DataFrame(
            test_values,
            columns=feature_names,
            index=test_df.index,
        )

        train_poly = train_poly.drop(
            columns=cols,
            errors="ignore",
        )

        test_poly = test_poly.drop(
            columns=cols,
            errors="ignore",
        )

        train_new = pd.concat(
            [
                train_df.drop(
                    columns=cols
                ),
                train_poly,
            ],
            axis=1,
        )

        test_new = pd.concat(
            [
                test_df.drop(
                    columns=cols
                ),
                test_poly,
            ],
            axis=1,
        )

        return (
            train_new,
            test_new,
            {
                "details": (
                    f"Polynomial degree={degree} fitted "
                    f"on train data only for {cols}"
                ),
                "columns_added": list(
                    train_poly.columns
                ),
                "columns_removed": cols,
            },
        )

    def _variance_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Select low-variance features using train data only."""

        numeric_train = train_df.select_dtypes(
            include=[np.number]
        )

        if numeric_train.empty:
            return (
                train_df.copy(),
                test_df.copy(),
                {
                    "details": (
                        "No numeric columns to check "
                        "variance, skipped"
                    ),
                    "columns_added": [],
                    "columns_removed": [],
                },
            )

        threshold = (step.params or {}).get(
            "threshold",
            0.01,
        )

        selector = VarianceThreshold(
            threshold=threshold
        )

        selector.fit(numeric_train)

        cols_to_drop = numeric_train.columns[
            ~selector.get_support()
        ].tolist()

        train_new = train_df.drop(
            columns=cols_to_drop,
            errors="ignore",
        )

        test_new = test_df.drop(
            columns=cols_to_drop,
            errors="ignore",
        )

        return (
            train_new,
            test_new,
            {
                "details": (
                    "Dropped low-variance columns "
                    "based on train data: "
                    f"{cols_to_drop}"
                ),
                "columns_added": [],
                "columns_removed": cols_to_drop,
            },
        )

    def _correlation_train_test(
        self,
        train_df: pd.DataFrame,
        test_df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Select highly correlated features using train data only."""

        numeric_train = train_df.select_dtypes(
            include=[np.number]
        )

        if numeric_train.shape[1] < 2:
            return (
                train_df.copy(),
                test_df.copy(),
                {
                    "details": (
                        "Less than 2 numeric columns, "
                        "nothing to drop"
                    ),
                    "columns_added": [],
                    "columns_removed": [],
                },
            )

        threshold = (step.params or {}).get(
            "threshold",
            0.90,
        )

        corr_matrix = numeric_train.corr().abs()

        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1,
            ).astype(bool)
        )

        to_drop = [
            column
            for column in upper.columns
            if any(
                upper[column] > threshold
            )
        ]

        train_new = train_df.drop(
            columns=to_drop,
            errors="ignore",
        )

        test_new = test_df.drop(
            columns=to_drop,
            errors="ignore",
        )

        return (
            train_new,
            test_new,
            {
                "details": (
                    "Dropped highly correlated "
                    "columns based on train data: "
                    f"{to_drop}"
                ),
                "columns_added": [],
                "columns_removed": to_drop,
            },
        )

    def _strip_target_column(
        self,
        step: FeatureEngineeringStep,
        target_column: str | None,
    ) -> tuple[FeatureEngineeringStep, bool]:
        """Return a sanitized copy of the step with target_column removed."""

        if (
            not target_column
            or target_column not in step.columns
        ):
            return step, False

        safe_columns = [
            c
            for c in step.columns
            if c != target_column
        ]

        return (
            step.model_copy(
                update={
                    "columns": safe_columns
                }
            ),
            True,
        )

    def _execute_step(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        """Dispatch to the correct action handler."""

        method_name = f"_action_{step.action}"

        method = getattr(
            self,
            method_name,
            None,
        )

        if method is None:
            raise ValueError(
                f"Unknown feature engineering action: "
                f"{step.action}"
            )

        return method(df, step)

    # =====================================================================
    # ACTION IMPLEMENTATIONS
    # Each follows: validate -> params -> execute -> track -> return
    # =====================================================================

    def _action_drop_columns(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        cols = [
            c
            for c in step.columns
            if c in df.columns
        ]

        new_df = (
            df.drop(columns=cols)
            if cols
            else df.copy()
        )

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": cols,
            "details": (
                f"Dropped {len(cols)} columns: {cols}"
                if cols
                else "No valid columns to drop"
            ),
        }

    def _action_one_hot_encode(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        cols = [
            c
            for c in step.columns
            if c in df.columns
        ]

        if not cols:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": (
                    "No valid columns found for "
                    "one-hot encoding, skipped"
                ),
            }

        drop_first = (step.params or {}).get(
            "drop_first",
            False,
        )

        new_df = pd.get_dummies(
            df,
            columns=cols,
            drop_first=drop_first,
        )

        added = [
            c
            for c in new_df.columns
            if c not in df.columns
        ]

        return {
            "dataframe": new_df,
            "columns_added": added,
            "columns_removed": [],
            "details": (
                f"One-hot encoded {cols}, "
                f"added {len(added)} columns"
            ),
        }

    def _action_label_encode(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        new_df = df.copy()
        added = []

        for col in step.columns:
            if col not in new_df.columns:
                continue

            new_col = f"{col}_encoded"

            new_df[new_col] = (
                new_df[col]
                .astype("category")
                .cat.codes
            )

            added.append(new_col)

        return {
            "dataframe": new_df,
            "columns_added": added,
            "columns_removed": [],
            "details": (
                f"Label encoded {step.columns} "
                f"into {added}"
                if added
                else "No valid columns found"
            ),
        }

    def _action_standard_scale(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        return self._scale(
            df,
            step,
            StandardScaler(),
            "Standard",
        )

    def _action_minmax_scale(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        return self._scale(
            df,
            step,
            MinMaxScaler(),
            "MinMax",
        )

    def _action_robust_scale(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        return self._scale(
            df,
            step,
            RobustScaler(),
            "Robust",
        )

    def _scale(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
        scaler,
        name: str,
    ):
        cols = [
            c
            for c in step.columns
            if (
                c in df.columns
                and pd.api.types.is_numeric_dtype(
                    df[c]
                )
            )
        ]

        if not cols:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": (
                    "No numeric columns found "
                    "to scale, skipped"
                ),
            }

        new_df = df.copy()

        scaled_values = scaler.fit_transform(
            new_df[cols]
        )

        for i, col in enumerate(cols):
            new_df[col] = scaled_values[:, i]

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": (
                f"{name} scaled {cols}"
            ),
        }

    def _action_log_transform(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        new_df = df.copy()

        cols = [
            c
            for c in step.columns
            if (
                c in new_df.columns
                and pd.api.types.is_numeric_dtype(
                    new_df[c]
                )
            )
        ]

        for col in cols:
            new_df[col] = np.log1p(
                new_df[col].clip(lower=0)
            )

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": (
                f"Log1p transformed {cols}"
                if cols
                else "No numeric columns found"
            ),
        }

    def _action_sqrt_transform(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        new_df = df.copy()

        cols = [
            c
            for c in step.columns
            if (
                c in new_df.columns
                and pd.api.types.is_numeric_dtype(
                    new_df[c]
                )
            )
        ]

        for col in cols:
            new_df[col] = np.sqrt(
                new_df[col].clip(lower=0)
            )

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": (
                f"Sqrt transformed {cols}"
                if cols
                else "No numeric columns found"
            ),
        }

    def _action_power_transform(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        new_df = df.copy()

        power = (step.params or {}).get(
            "power",
            2,
        )

        cols = [
            c
            for c in step.columns
            if (
                c in new_df.columns
                and pd.api.types.is_numeric_dtype(
                    new_df[c]
                )
            )
        ]

        for col in cols:
            new_df[col] = np.power(
                new_df[col],
                power,
            )

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": [],
            "details": (
                f"Power transformed {cols} "
                f"with power={power}"
                if cols
                else "No numeric columns found"
            ),
        }

    def _action_binning(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        new_df = df.copy()

        bins = (step.params or {}).get(
            "bins",
            5,
        )

        strategy = (step.params or {}).get(
            "strategy",
            "quantile",
        )

        added = []

        for col in step.columns:
            if (
                col not in new_df.columns
                or not pd.api.types.is_numeric_dtype(
                    new_df[col]
                )
            ):
                continue

            new_col = f"{col}_binned"

            if strategy == "quantile":
                new_df[new_col] = pd.qcut(
                    new_df[col],
                    q=bins,
                    labels=False,
                    duplicates="drop",
                )
            else:
                new_df[new_col] = pd.cut(
                    new_df[col],
                    bins=bins,
                    labels=False,
                )

            added.append(new_col)

        return {
            "dataframe": new_df,
            "columns_added": added,
            "columns_removed": [],
            "details": (
                f"Binned {step.columns} into "
                f"{bins} bins ({strategy})"
                if added
                else "No valid numeric columns"
            ),
        }

    def _action_create_interaction(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        new_df = df.copy()

        cols = [
            c
            for c in step.columns
            if c in new_df.columns
        ]

        if len(cols) < 2:
            return {
                "dataframe": new_df,
                "columns_added": [],
                "columns_removed": [],
                "details": (
                    "Need at least 2 columns for "
                    "interaction, skipped"
                ),
            }

        operation = (step.params or {}).get(
            "operation",
            "multiply",
        )

        ops = {
            "multiply": lambda a, b: a * b,
            "divide": lambda a, b: a / (b + 1e-9),
            "add": lambda a, b: a + b,
            "subtract": lambda a, b: a - b,
        }

        op_func = ops.get(
            operation,
            ops["multiply"],
        )

        new_col = (
            f"{'_'.join(cols[:2])}_{operation}"
        )

        new_df[new_col] = op_func(
            new_df[cols[0]],
            new_df[cols[1]],
        )

        return {
            "dataframe": new_df,
            "columns_added": [new_col],
            "columns_removed": [],
            "details": (
                f"Created interaction column "
                f"{new_col}"
            ),
        }

    def _action_create_polynomial(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        cols = [
            c
            for c in step.columns
            if (
                c in df.columns
                and pd.api.types.is_numeric_dtype(
                    df[c]
                )
            )
        ]

        if not cols:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": (
                    "No numeric columns found for "
                    "polynomial features, skipped"
                ),
            }

        degree = (step.params or {}).get(
            "degree",
            2,
        )

        include_bias = (step.params or {}).get(
            "include_bias",
            False,
        )

        poly = PolynomialFeatures(
            degree=degree,
            include_bias=include_bias,
        )

        poly_features = poly.fit_transform(
            df[cols]
        )

        feature_names = poly.get_feature_names_out(
            cols
        )

        poly_df = pd.DataFrame(
            poly_features,
            columns=feature_names,
            index=df.index,
        )

        poly_df = poly_df.drop(
            columns=cols,
            errors="ignore",
        )

        new_df = pd.concat(
            [
                df.drop(columns=cols),
                poly_df,
            ],
            axis=1,
        )

        return {
            "dataframe": new_df,
            "columns_added": list(
                poly_df.columns
            ),
            "columns_removed": cols,
            "details": (
                f"Polynomial degree={degree} for "
                f"{cols}, generated "
                f"{len(poly_df.columns)} features"
            ),
        }

    def _action_drop_low_variance(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        if numeric_df.empty:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": (
                    "No numeric columns to check "
                    "variance, skipped"
                ),
            }

        threshold = (step.params or {}).get(
            "threshold",
            0.01,
        )

        selector = VarianceThreshold(
            threshold=threshold
        )

        selector.fit(numeric_df)

        cols_to_keep = numeric_df.columns[
            selector.get_support()
        ].tolist()

        cols_to_drop = numeric_df.columns[
            ~selector.get_support()
        ].tolist()

        non_numeric = df.select_dtypes(
            exclude=[np.number]
        ).columns.tolist()

        new_df = df[
            non_numeric + cols_to_keep
        ]

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": cols_to_drop,
            "details": (
                f"Dropped low-variance columns: "
                f"{cols_to_drop}"
                if cols_to_drop
                else "No low-variance columns found"
            ),
        }

    def _action_drop_high_correlation(
        self,
        df: pd.DataFrame,
        step: FeatureEngineeringStep,
    ):
        numeric_df = df.select_dtypes(
            include=[np.number]
        )

        if numeric_df.shape[1] < 2:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": (
                    "Less than 2 numeric columns, "
                    "nothing to drop"
                ),
            }

        threshold = (step.params or {}).get(
            "threshold",
            0.90,
        )

        corr_matrix = numeric_df.corr().abs()

        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1,
            ).astype(bool)
        )

        to_drop = [
            column
            for column in upper.columns
            if any(
                upper[column] > threshold
            )
        ]

        if not to_drop:
            return {
                "dataframe": df.copy(),
                "columns_added": [],
                "columns_removed": [],
                "details": (
                    f"No correlations above "
                    f"{threshold}, nothing dropped"
                ),
            }

        new_df = df.drop(
            columns=to_drop
        )

        return {
            "dataframe": new_df,
            "columns_added": [],
            "columns_removed": to_drop,
            "details": (
                f"Dropped highly correlated "
                f"columns (|r| > {threshold}): "
                f"{to_drop}"
            ),
        }
