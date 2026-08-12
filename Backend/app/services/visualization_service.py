import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from schema.visualization_plan import (
    VisualizationPlan,
    VisualizationPlanResponse,
)


class VisualizationService:

    def generate_visualizations(
        self,
        dataframe: pd.DataFrame,
        visualization_plan: VisualizationPlanResponse,
    ):

        results = []

        for plan in visualization_plan.visualizations:

            if plan.chart_type.value == "scatter":
                result = self.generate_scatter_plot(
                    dataframe,
                    plan
                )

            if plan.chart_type.value == "bar":
                result = self.generate_bar_chart(
                    dataframe,
                    plan
                )
            if plan.chart_type.value == "histogram":
                result = self.generate_histogram(
                    dataframe,
                    plan
                )
            if plan.chart_type.value == "box":
                result = self.generate_box_plot(
                    dataframe,
                    plan
                )
            if plan.chart_type.value == "heatmap":
                result = self.generate_heatmap(
                    dataframe,
                    plan
                )

            results.append(result)

        return results

    def generate_bar_chart(
        self,
        dataframe: pd.DataFrame,
        visualization_plan: VisualizationPlanResponse,
    ):
        ...

    def generate_histogram(
        self,
        dataframe: pd.DataFrame,
        visualization_plan: VisualizationPlanResponse,
    ):
        ...

    def generate_scatter_plot(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
    ):

        if not plan.x_column:
            raise ValueError("Scatter plot requires an x column.")

        if not plan.y_column:
            raise ValueError("Scatter plot requires a y column.")

        if plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe."
            )

        if plan.y_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.y_column}' not found in dataframe."
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

        output_dir = Path("outputs/charts")
        output_dir.mkdir(parents=True, exist_ok=True)

        filename = (
            f"{plan.x_column}_vs_{plan.y_column}_scatter.png"
        )

        output_path = output_dir / filename

        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "scatter",
            "x_column": plan.x_column,
            "y_column": plan.y_column,
            "group_by": plan.group_by,
            "path": str(output_path),
        }

    def generate_box_plot(
        self,
        dataframe: pd.DataFrame,
        visualization_plan: VisualizationPlanResponse,
    ):
        ...

    def generate_heatmap(
        self,
        dataframe: pd.DataFrame,
        visualization_plan: VisualizationPlanResponse,
    ):
        ...


visualization_service = VisualizationService()
