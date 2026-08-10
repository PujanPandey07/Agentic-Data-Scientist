import pandas as pd


class CleaningService:

    def clean_dataset(self, dataframe: pd.DataFrame):
        original_rows = len(dataframe)

        dataframe = self.remove_duplicates(dataframe)
        dataframe = self.handle_missing_values(dataframe)
        dataframe = self.standardize_column_names(dataframe)

        report = {
            "original_rows": original_rows,
            "final_rows": len(dataframe),
            "duplicates_removed": original_rows - len(dataframe),
            "missing_values_remaining": int(dataframe.isnull().sum().sum()),
        }

        return dataframe, report

    def remove_duplicates(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        return dataframe.drop_duplicates().reset_index(drop=True)

    def handle_missing_values(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        dataframe = dataframe.copy()

        numerical_columns = dataframe.select_dtypes(
            include="number"
        ).columns

        categorical_columns = dataframe.select_dtypes(
            include=["object", "category"]
        ).columns

        for column in numerical_columns:
            if dataframe[column].isna().any():
                dataframe[column] = dataframe[column].fillna(
                    dataframe[column].median()
                )

        for column in categorical_columns:
            if dataframe[column].isna().any():
                mode = dataframe[column].mode()

                if not mode.empty:
                    dataframe[column] = dataframe[column].fillna(
                        mode.iloc[0]
                    )
                else:
                    dataframe[column] = dataframe[column].fillna(
                        "Unknown"
                    )

        return dataframe

    def standardize_column_names(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        dataframe = dataframe.copy()

        dataframe.columns = (
            dataframe.columns
            .str.strip()
            .str.lower()
            .str.replace(" ", "_")
        )

        return dataframe


cleaning_service = CleaningService()
