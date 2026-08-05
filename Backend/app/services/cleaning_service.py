
import pandas as pd


class CleaningService:

    def clean_dataset(self, dataframe: pd.DataFrame) -> pd.DataFrame:

        dataframe = dataframe.drop_duplicates()

        return dataframe
