import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from pathlib import Path

from schema.visualization_plan import (
    VisualizationPlan,
    VisualizationPlanResponse,
)

logger = logging.getLogger(__name__)


class VisualizationService:

    def __init__(self):
        self.output_dir = Path("outputs/charts")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------
    # Main dispatcher
    # ---------------------------------------------------------

    def generate_visualizations(
        self,
        dataframe: pd.DataFrame,
        visualization_plan: VisualizationPlanResponse,
    ):
        logger.info(
            f"Starting visualization generation: {len(visualization_plan.visualizations)} charts planned"
        )

        results = []
        failures = []

        for plan in visualization_plan.visualizations:

            chart_type = plan.chart_type.value

            try:
                if chart_type == "scatter":
                    result = self.generate_scatter_plot(dataframe, plan)
                elif chart_type == "bar":
                    result = self.generate_bar_chart(dataframe, plan)
                elif chart_type == "histogram":
                    result = self.generate_histogram(dataframe, plan)
                elif chart_type == "box":
                    result = self.generate_box_plot(dataframe, plan)
                elif chart_type == "heatmap":
                    result = self.generate_heatmap(dataframe, plan)
                else:
                    raise ValueError(f"Unsupported chart type: {chart_type}")

            except Exception as e:
                # Fault isolation: one bad chart shouldn't take down the
                # other N-1 that were perfectly valid. Same pattern as
                # FeatureEngineeringService.apply_plan.
                logger.warning(
                    f"Chart '{chart_type}' (x={plan.x_column}, "
                    f"y={plan.y_column}) failed, skipped: {e}"
                )
                failures.append({
                    "chart_type": chart_type,
                    "x_column": plan.x_column,
                    "y_column": plan.y_column,
                    "error": str(e),
                })
                continue

            logger.info(f"Generated {chart_type} chart -> {result['path']}")
            results.append(result)

        if failures:
            logger.warning(
                f"Visualization generation done with {len(failures)} "
                f"failure(s): {[f['chart_type'] for f in failures]}"
            )

        logger.info(
            f"Visualization generation done: {len(results)} succeeded, "
            f"{len(failures)} failed"
        )

        return results

    # ---------------------------------------------------------
    # Bar chart
    # ---------------------------------------------------------

    def generate_bar_chart(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
    ):

        if not plan.x_column:
            raise ValueError(
                "Bar chart requires an x column."
            )

        if plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe."
            )

        if plan.group_by and plan.group_by not in dataframe.columns:
            raise ValueError(
                f"Group-by column '{plan.group_by}' "
                f"not found in dataframe."
            )

        plt.figure(figsize=(8, 6))

        if plan.group_by:

            sns.countplot(
                data=dataframe,
                x=plan.x_column,
                hue=plan.group_by,
            )

        else:

            sns.countplot(
                data=dataframe,
                x=plan.x_column,
            )

        plt.title(
            f"Count of {plan.x_column}"
        )

        plt.tight_layout()

        filename = (
            f"{plan.x_column}_bar_chart.png"
        )

        output_path = self.output_dir / filename

        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "bar",
            "x_column": plan.x_column,
            "group_by": plan.group_by,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Histogram
    # ---------------------------------------------------------

    def generate_histogram(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
    ):

        if not plan.x_column:
            raise ValueError(
                "Histogram requires an x column."
            )

        if plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe."
            )

        plt.figure(figsize=(8, 6))

        sns.histplot(
            data=dataframe,
            x=plan.x_column,
            kde=True,
        )

        plt.title(
            f"Distribution of {plan.x_column}"
        )

        plt.tight_layout()

        filename = (
            f"{plan.x_column}_histogram.png"
        )

        output_path = self.output_dir / filename

        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "histogram",
            "x_column": plan.x_column,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Scatter plot
    # ---------------------------------------------------------

    def generate_scatter_plot(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
    ):

        if not plan.x_column:
            raise ValueError(
                "Scatter plot requires an x column."
            )

        if not plan.y_column:
            raise ValueError(
                "Scatter plot requires a y column."
            )

        if plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe."
            )

        if plan.y_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.y_column}' not found in dataframe."
            )

        if plan.group_by and plan.group_by not in dataframe.columns:
            raise ValueError(
                f"Group-by column '{plan.group_by}' "
                f"not found in dataframe."
            )

        plt.figure(figsize=(8, 6))

        if plan.group_by:

            sns.scatterplot(
                data=dataframe,
                x=plan.x_column,
                y=plan.y_column,
                hue=plan.group_by,
            )

        else:

            sns.scatterplot(
                data=dataframe,
                x=plan.x_column,
                y=plan.y_column,
            )

        plt.title(
            f"{plan.x_column} vs {plan.y_column}"
        )

        plt.tight_layout()

        filename = (
            f"{plan.x_column}_vs_"
            f"{plan.y_column}_scatter.png"
        )

        output_path = self.output_dir / filename

        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "scatter",
            "x_column": plan.x_column,
            "y_column": plan.y_column,
            "group_by": plan.group_by,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Box plot
    # ---------------------------------------------------------

    def generate_box_plot(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
    ):

        if not plan.y_column:
            raise ValueError(
                "Box plot requires a y column."
            )

        if plan.y_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.y_column}' not found in dataframe."
            )

        if plan.x_column and plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe."
            )

        plt.figure(figsize=(8, 6))

        if plan.x_column:

            sns.boxplot(
                data=dataframe,
                x=plan.x_column,
                y=plan.y_column,
            )

        else:

            sns.boxplot(
                data=dataframe,
                y=plan.y_column,
            )

        plt.title(
            f"Distribution of {plan.y_column}"
        )

        plt.tight_layout()

        if plan.x_column:

            filename = (
                f"{plan.y_column}_by_"
                f"{plan.x_column}_box.png"
            )

        else:

            filename = (
                f"{plan.y_column}_box.png"
            )

        output_path = self.output_dir / filename

        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "box",
            "x_column": plan.x_column,
            "y_column": plan.y_column,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Heatmap
    # ---------------------------------------------------------

    def generate_heatmap(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
    ):

        numerical_dataframe = dataframe.select_dtypes(
            include="number"
        )

        if numerical_dataframe.empty:
            raise ValueError(
                "Heatmap requires numerical columns."
            )

        correlation_matrix = (
            numerical_dataframe.corr()
        )

        plt.figure(figsize=(10, 8))

        sns.heatmap(
            correlation_matrix,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            center=0,
        )

        plt.title(
            "Feature Correlation Heatmap"
        )

        plt.tight_layout()

        filename = "correlation_heatmap.png"

        output_path = self.output_dir / filename

        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "heatmap",
            "columns": list(
                numerical_dataframe.columns
            ),
            "path": str(output_path),
        }


visualization_service = VisualizationService()
