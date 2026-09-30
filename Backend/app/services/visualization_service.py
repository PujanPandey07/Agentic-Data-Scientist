# services/visualization_service.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from pathlib import Path
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram as scipy_dendrogram

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
        dataset_id: str,
    ):
        logger.info(
            f"Starting visualization generation: {len(visualization_plan.visualizations)} charts planned"
        )

        # Namespace every chart under this dataset's own subfolder, so two
        # different datasets sharing a column name (e.g. "age") never
        # overwrite each other's chart files on disk.
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        results = []
        failures = []

        for plan in visualization_plan.visualizations:

            chart_type = plan.chart_type.value

            try:
                if chart_type == "scatter":
                    result = self.generate_scatter_plot(
                        dataframe, plan, dataset_dir)
                elif chart_type == "bar":
                    result = self.generate_bar_chart(
                        dataframe, plan, dataset_dir)
                elif chart_type == "histogram":
                    result = self.generate_histogram(
                        dataframe, plan, dataset_dir)
                elif chart_type == "box":
                    result = self.generate_box_plot(
                        dataframe, plan, dataset_dir)
                elif chart_type == "heatmap":
                    result = self.generate_heatmap(
                        dataframe, plan, dataset_dir)
                elif chart_type == "line":
                    result = self.generate_line_chart(
                        dataframe, plan, dataset_dir)
                elif chart_type == "violin":
                    result = self.generate_violin_plot(
                        dataframe, plan, dataset_dir)
                elif chart_type == "pairplot":
                    result = self.generate_pairplot(
                        dataframe, plan, dataset_dir)
                elif chart_type == "pie":
                    result = self.generate_pie_chart(
                        dataframe, plan, dataset_dir)
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
        output_dir: Path,
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

        output_path = output_dir / filename

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
        output_dir: Path,
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

        output_path = output_dir / filename

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
        output_dir: Path,
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

    # ---------------------------------------------------------
    # Box plot
    # ---------------------------------------------------------

    def generate_box_plot(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
        output_dir: Path,
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

        output_path = output_dir / filename

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
        output_dir: Path,
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

        output_path = output_dir / filename

        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "heatmap",
            "columns": list(
                numerical_dataframe.columns
            ),
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Line chart
    # ---------------------------------------------------------

    def generate_line_chart(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
        output_dir: Path,
    ):
        if not plan.x_column:
            raise ValueError("Line chart requires an x column.")
        if not plan.y_column:
            raise ValueError("Line chart requires a y column.")
        if plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe.")
        if plan.y_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.y_column}' not found in dataframe.")
        if plan.group_by and plan.group_by not in dataframe.columns:
            raise ValueError(
                f"Group-by column '{plan.group_by}' not found in dataframe.")

        # Line charts need the x-axis sorted, or the connecting line
        # visually jumps around meaninglessly.
        plot_df = dataframe.sort_values(by=plan.x_column)

        plt.figure(figsize=(8, 6))

        if plan.group_by:
            sns.lineplot(data=plot_df, x=plan.x_column,
                         y=plan.y_column, hue=plan.group_by)
        else:
            sns.lineplot(data=plot_df, x=plan.x_column, y=plan.y_column)

        plt.title(f"{plan.y_column} over {plan.x_column}")
        plt.tight_layout()

        filename = f"{plan.y_column}_over_{plan.x_column}_line.png"
        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "line",
            "x_column": plan.x_column,
            "y_column": plan.y_column,
            "group_by": plan.group_by,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Violin plot
    # ---------------------------------------------------------

    def generate_violin_plot(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
        output_dir: Path,
    ):
        if not plan.y_column:
            raise ValueError("Violin plot requires a y column.")
        if plan.y_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.y_column}' not found in dataframe.")
        if plan.x_column and plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe.")

        plt.figure(figsize=(8, 6))

        if plan.x_column:
            sns.violinplot(data=dataframe, x=plan.x_column, y=plan.y_column)
        else:
            sns.violinplot(data=dataframe, y=plan.y_column)

        plt.title(f"Distribution shape of {plan.y_column}")
        plt.tight_layout()

        if plan.x_column:
            filename = f"{plan.y_column}_by_{plan.x_column}_violin.png"
        else:
            filename = f"{plan.y_column}_violin.png"

        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "violin",
            "x_column": plan.x_column,
            "y_column": plan.y_column,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Pairplot
    # ---------------------------------------------------------

    def generate_pairplot(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
        output_dir: Path,
    ):
        numerical_dataframe = dataframe.select_dtypes(include="number")

        if numerical_dataframe.shape[1] < 2:
            raise ValueError("Pairplot requires at least 2 numerical columns.")

        # Cap the number of columns fed in — an accidental pairplot over
        # dozens of numeric columns is both unreadable and extremely slow
        # to render (O(n^2) subplots). If the plan's group_by names a real
        # categorical column, use it as hue; otherwise plot ungrouped.
        cols = list(numerical_dataframe.columns)[:8]
        plot_df = dataframe[cols].copy()

        hue = None
        if plan.group_by and plan.group_by in dataframe.columns:
            hue = plan.group_by
            plot_df[hue] = dataframe[hue]

        grid = sns.pairplot(plot_df, hue=hue)
        grid.figure.suptitle("Pairwise relationships", y=1.02)

        filename = "pairplot.png"
        output_path = output_dir / filename
        grid.savefig(output_path)
        plt.close(grid.figure)

        return {
            "chart_type": "pairplot",
            "columns": cols,
            "group_by": hue,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Pie chart
    # ---------------------------------------------------------

    def generate_pie_chart(
        self,
        dataframe: pd.DataFrame,
        plan: VisualizationPlan,
        output_dir: Path,
    ):
        if not plan.x_column:
            raise ValueError("Pie chart requires an x column.")
        if plan.x_column not in dataframe.columns:
            raise ValueError(
                f"Column '{plan.x_column}' not found in dataframe.")

        counts = dataframe[plan.x_column].value_counts()

        # A pie chart with too many slices is unreadable — this is a
        # deterministic safety net in case the LLM planner picks a
        # higher-cardinality column than intended despite the prompt's
        # guidance to avoid that.
        if len(counts) > 10:
            raise ValueError(
                f"Column '{plan.x_column}' has {len(counts)} unique values — "
                f"too many for a readable pie chart (limit: 10)."
            )

        plt.figure(figsize=(8, 8))
        plt.pie(counts.values, labels=counts.index.astype(
            str), autopct="%1.1f%%")
        plt.title(f"Proportion of {plan.x_column}")
        plt.tight_layout()

        filename = f"{plan.x_column}_pie.png"
        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "pie",
            "x_column": plan.x_column,
            "path": str(output_path),
        }

    # ---------------------------------------------------------
    # Clustering-only visualizations — deterministic, not LLM-planned.
    # Called directly from graphs/nodes.py once cluster_labels/
    # elbow_curve/linkage_matrix exist (after tuning), not through the
    # normal visualization_plan dispatch above.
    # ---------------------------------------------------------

    def generate_clustering_visualizations(
        self,
        dataframe: pd.DataFrame,
        cluster_labels: list[int] | None,
        elbow_curve: list[dict] | None,
        linkage_matrix: list | None,
        dataset_id: str,
    ) -> list[dict]:
        """Orchestrator: generates whichever of the three clustering charts
        have the data to support them. Same fault-isolation pattern as
        generate_visualizations — one failing chart doesn't block the
        others."""
        dataset_dir = self.output_dir / dataset_id
        dataset_dir.mkdir(parents=True, exist_ok=True)

        results = []

        if cluster_labels:
            try:
                results.append(self.generate_cluster_scatter(
                    dataframe, cluster_labels, dataset_dir))
            except Exception as e:
                logger.warning(f"Cluster scatter plot failed, skipped: {e}")

        if elbow_curve:
            try:
                results.append(self.generate_elbow_chart(
                    elbow_curve, dataset_dir))
            except Exception as e:
                logger.warning(f"Elbow chart failed, skipped: {e}")

        if linkage_matrix:
            try:
                results.append(self.generate_dendrogram(
                    linkage_matrix, dataset_dir))
            except Exception as e:
                logger.warning(f"Dendrogram failed, skipped: {e}")

        return results

    def generate_cluster_scatter(
        self,
        dataframe: pd.DataFrame,
        cluster_labels: list[int],
        output_dir: Path,
    ):
        """Projects the dataset to 2D via PCA (if more than 2 numeric
        columns) purely for plotting, then colors points by their
        assigned cluster label. -1 (DBSCAN noise) is shown as a distinct
        'noise' category rather than silently mixed into the color scale."""
        numeric_df = dataframe.select_dtypes(include="number")

        if numeric_df.shape[1] < 2:
            raise ValueError(
                "Cluster scatter plot requires at least 2 numeric columns."
            )

        if len(cluster_labels) != len(numeric_df):
            raise ValueError(
                f"Label count ({len(cluster_labels)}) doesn't match "
                f"dataframe row count ({len(numeric_df)})."
            )

        if numeric_df.shape[1] > 2:
            coords = PCA(n_components=2, random_state=42).fit_transform(
                numeric_df.values)
            x_label, y_label = "PC1", "PC2"
        else:
            coords = numeric_df.values
            x_label, y_label = numeric_df.columns[0], numeric_df.columns[1]

        plot_df = pd.DataFrame({
            x_label: coords[:, 0],
            y_label: coords[:, 1],
            "cluster": ["noise" if l == -1 else str(l) for l in cluster_labels],
        })

        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=plot_df, x=x_label, y=y_label,
                        hue="cluster", palette="tab10")
        plt.title("Cluster assignments" +
                  (" (PCA-projected)" if numeric_df.shape[1] > 2 else ""))
        plt.tight_layout()

        filename = "cluster_scatter.png"
        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "cluster_scatter",
            "x_column": x_label,
            "y_column": y_label,
            "path": str(output_path),
        }

    def generate_elbow_chart(
        self,
        elbow_curve: list[dict],
        output_dir: Path,
    ):
        """elbow_curve is a list of {n_clusters, score, inertia} points,
        already sorted by n_clusters (see ClusteringTuningService.tune).
        Plots the tuning objective's score against n_clusters — the
        classic 'elbow' the user looks for."""
        if not elbow_curve:
            raise ValueError("No elbow curve data available to plot.")

        n_values = [p["n_clusters"] for p in elbow_curve]
        scores = [p["score"] for p in elbow_curve]

        plt.figure(figsize=(8, 6))
        plt.plot(n_values, scores, marker="o")
        plt.xlabel("Number of clusters")
        plt.ylabel("Clustering quality score")
        plt.title("Elbow / knee diagram")
        plt.xticks(n_values)
        plt.tight_layout()

        filename = "elbow_diagram.png"
        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "elbow",
            "path": str(output_path),
        }

    def generate_dendrogram(
        self,
        linkage_matrix: list,
        output_dir: Path,
    ):
        """linkage_matrix is the list-serialized form of scipy's
        linkage() output (see ClusteringTuningService.tune, which stores
        it via .tolist() for JSON/Redis transport) — converted back to an
        array here since scipy's dendrogram() needs the real ndarray."""
        if not linkage_matrix:
            raise ValueError("No linkage matrix available to plot.")

        Z = np.array(linkage_matrix)

        plt.figure(figsize=(10, 6))
        scipy_dendrogram(Z, truncate_mode="lastp", p=30, leaf_rotation=90.)
        plt.title("Hierarchical clustering dendrogram")
        plt.xlabel("Sample index (or cluster size)")
        plt.ylabel("Distance")
        plt.tight_layout()

        filename = "dendrogram.png"
        output_path = output_dir / filename
        plt.savefig(output_path)
        plt.close()

        return {
            "chart_type": "dendrogram",
            "path": str(output_path),
        }


visualization_service = VisualizationService()
