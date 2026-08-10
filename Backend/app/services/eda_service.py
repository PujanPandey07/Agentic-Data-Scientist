from typing import Any
import pandas as pd
from graphs.state import GraphState


class EDAService:

    def analyze(self, dataframe: pd.DataFrame) -> dict[str, Any]:
        """
        Perform basic exploratory data analysis on a DataFrame.

        The service is responsible only for analyzing the data.
        It does not modify GraphState and does not load files.
        """

        if dataframe is None or dataframe.empty:
            raise ValueError("Cannot perform EDA on an empty dataset.")

        numerical_df = dataframe.select_dtypes(include="number")
        categorical_df = dataframe.select_dtypes(
            include=["object", "category", "bool"]
        )

        numerical_summary = {}

        if not numerical_df.empty:
            numerical_summary = (
                numerical_df
                .describe()
                .round(4)
                .to_dict()
            )

        categorical_summary = {}

        for column in categorical_df.columns:
            categorical_summary[column] = {
                "unique_values": int(dataframe[column].nunique(dropna=True)),
                "value_counts": (
                    dataframe[column]
                    .value_counts(dropna=False)
                    .to_dict()
                ),
            }

        correlation = {}

        if numerical_df.shape[1] >= 2:
            correlation = (
                numerical_df
                .corr()
                .round(4)
                .to_dict()
            )

        eda_report = {
            "shape": {
                "rows": len(dataframe),
                "columns": len(dataframe.columns),
            },
            "column_names": list(dataframe.columns),
            "numerical_columns": list(numerical_df.columns),
            "categorical_columns": list(categorical_df.columns),
            "numerical_summary": numerical_summary,
            "categorical_summary": categorical_summary,
            "missing_values": dataframe.isnull().sum().to_dict(),
            "correlation": correlation,
        }

        return eda_report


eda_service = EDAService()
