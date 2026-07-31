import pandas as pd

from schema.dataset_summary import DatasetSummary


class DatasetInspector:

    def inspect(self, dataframe: pd.DataFrame) -> DatasetSummary:

        missing = dataframe.isnull().sum().to_dict()

        numerical = dataframe.select_dtypes(include="number").columns.tolist()

        categorical = dataframe.select_dtypes(
            exclude="number").columns.tolist()

        memory = dataframe.memory_usage(deep=True).sum()

        return DatasetSummary(
            rows=len(dataframe),
            columns=len(dataframe.columns),

            column_names=dataframe.columns.tolist(),

            data_types={
                column: str(dtype)
                for column, dtype in dataframe.dtypes.items()
            },

            numerical_columns=numerical,

            categorical_columns=categorical,

            missing_values=missing,

            duplicate_rows=int(dataframe.duplicated().sum()),

            memory_usage=f"{memory / 1024:.2f} KB",

            has_missing_values=any(value > 0 for value in missing.values()),

            has_duplicates=dataframe.duplicated().any(),

            potential_target_columns=[],
        )


dataset_inspector = DatasetInspector()
