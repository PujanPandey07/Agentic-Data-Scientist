import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CleaningService:

    def clean_dataset(self, dataframe: pd.DataFrame):
        original_rows = len(dataframe)
        logger.info(
            f"Starting cleaning: {original_rows} rows, {len(dataframe.columns)} columns")

        dataframe = self.remove_duplicates(dataframe)
        dataframe = self.handle_missing_values(dataframe)
        dataframe = self.standardize_column_names(dataframe)

        report = {
            "original_rows": original_rows,
            "final_rows": len(dataframe),
            "duplicates_removed": original_rows - len(dataframe),
            "missing_values_remaining": int(dataframe.isnull().sum().sum()),
        }
        logger.info(f"Cleaning done: {report}")

        return dataframe, report

    def remove_duplicates(
        self,
        dataframe: pd.DataFrame
    ) -> pd.DataFrame:

        before = len(dataframe)
        dataframe = dataframe.drop_duplicates().reset_index(drop=True)
        logger.info(f"Removed {before - len(dataframe)} duplicate rows")

        return dataframe

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
                logger.info(f"Filled missing values in '{column}' with median")

        for column in categorical_columns:
            if dataframe[column].isna().any():
                mode = dataframe[column].mode()

                if not mode.empty:
                    dataframe[column] = dataframe[column].fillna(
                        mode.iloc[0]
                    )
                    logger.info(
                        f"Filled missing values in '{column}' with mode")
                else:
                    dataframe[column] = dataframe[column].fillna(
                        "Unknown"
                    )
                    logger.info(
                        f"Filled missing values in '{column}' with 'Unknown'")

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
