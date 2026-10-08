import pandas as pd
import logging

logger = logging.getLogger(__name__)


class CleaningService:

    def clean_dataset(self, dataframe: pd.DataFrame):
        original_rows = len(dataframe)
        logger.info(
            f"Starting cleaning: {original_rows} rows, {len(dataframe.columns)} columns")

        dataframe = self.remove_duplicates(dataframe)
        dataframe = self.auto_parse_datetimes(dataframe)
        dataframe = self.auto_coerce_numerics(dataframe)
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

    def auto_parse_datetimes(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Probe string/object columns: if values parse into valid dates, convert them.

        Guard against false-positives: skip columns that look like plain integers
        (no separators such as '-', '/', ' ') even if they technically parse — these
        are more likely year-integers or ID numbers than real date strings.
        """
        dataframe = dataframe.copy()
        candidate_cols = dataframe.select_dtypes(include=["object"]).columns

        hints = {"date", "time", "timestamp", "period", "datetime", "year", "month", "day"}
        for col in candidate_cols:
            is_hinted = any(h in col.lower() for h in hints)
            series = dataframe[col].dropna()
            if series.empty:
                continue

            # Skip columns that look like bare integers (no date separators)
            sample_str = series.head(50).astype(str)
            has_separator = sample_str.str.contains(r"[-/:\s]", regex=True).mean()
            if has_separator < 0.5 and not is_hinted:
                continue

            sample = series.head(50)
            try:
                parsed_sample = pd.to_datetime(sample, errors="coerce")
                valid_ratio = parsed_sample.notna().mean()
                # If high ratio or hinted and mostly parseable, convert column
                if valid_ratio >= 0.8 or (is_hinted and valid_ratio >= 0.5):
                    dataframe[col] = pd.to_datetime(dataframe[col], errors="coerce")
                    logger.info(
                        f"Automatically converted '{col}' to datetime "
                        f"({valid_ratio*100:.1f}% valid)"
                    )
            except Exception as e:
                logger.debug(f"Could not convert column '{col}' to datetime: {e}")

        return dataframe

    def auto_coerce_numerics(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Probe string/object columns: if values become valid numbers after
        stripping leading non-numeric characters (e.g. '?0.1' -> 0.1), and
        >= 70% of a sample parse successfully, coerce the whole column to float.

        Guards:
        - Skips columns already converted to datetime64 (handled above).
        - Skips columns where stripping leaves mostly empty strings (true categorical).
        """
        dataframe = dataframe.copy()
        for col in dataframe.select_dtypes(include=["object"]).columns:
            # Skip if already datetime
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                continue
            series = dataframe[col].dropna()
            if series.empty:
                continue
            sample = series.head(100)
            # Strip any leading/trailing non-numeric chars (preserves '-', '.', 'e')
            stripped = (
                sample.astype(str)
                .str.strip()
                .str.replace(r"^[^0-9\-\.]+", "", regex=True)
                .str.replace(r"[^0-9\-\.eE]+$", "", regex=True)
            )
            # Skip if stripping left mostly empty strings (true categorical)
            if (stripped == "").mean() > 0.3:
                continue
            parsed = pd.to_numeric(stripped, errors="coerce")
            valid_ratio = parsed.notna().mean()
            if valid_ratio >= 0.7:
                # Apply coercion to the full column
                full_stripped = (
                    dataframe[col].astype(str)
                    .str.strip()
                    .str.replace(r"^[^0-9\-\.]+", "", regex=True)
                    .str.replace(r"[^0-9\-\.eE]+$", "", regex=True)
                )
                dataframe[col] = pd.to_numeric(full_stripped, errors="coerce")
                logger.info(
                    f"Auto-coerced '{col}' to numeric "
                    f"({valid_ratio*100:.1f}% valid after stripping non-numeric chars)"
                )
        return dataframe

    def remove_duplicates(
        self,
        dataframe: pd.DataFrame,
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

        dataframe.columns = [
            self.standardize_column_name(column)
            for column in dataframe.columns
        ]

        return dataframe

    def standardize_column_name(self, column: str) -> str:
        return column.strip().lower().replace(" ", "_")


cleaning_service = CleaningService()
