import pandas as pd

from app.models.dataset_profile import DatasetProfile


class DataProfiler:

    @staticmethod
    def generate_profile(df: pd.DataFrame):

        return DatasetProfile(

            rows=df.shape[0],

            columns=df.shape[1],

            column_names=list(df.columns),

            data_types=df.dtypes.astype(str).to_dict(),

            missing_values=df.isnull().sum().to_dict(),

            duplicate_rows=int(df.duplicated().sum()),

            memory_usage_mb=round(
                df.memory_usage(deep=True).sum()/1024**2,
                2
            ),

            numerical_columns=df.select_dtypes(
                include="number"
            ).columns.tolist(),

            categorical_columns=df.select_dtypes(
                include="object"
            ).columns.tolist()

        )