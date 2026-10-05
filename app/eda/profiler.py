import os
import re
from typing import Optional
import pandas as pd

from app.eda.column_normalizer import ColumnNormalizer
from app.models.dataset_profile import DatasetProfile


class DataProfiler:

    @staticmethod
    def generate_profile(df: pd.DataFrame, filename: Optional[str] = None) -> DatasetProfile:
        n_rows = df.shape[0]
        n_cols = df.shape[1]

        # Categorical columns (handle both 'object', 'category', and 'string' dtypes without pandas deprecation warning)
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns.tolist()

        # Datetime columns
        dt_cols = df.select_dtypes(include=["datetime", "datetime64"]).columns.tolist()

        # If not explicitly datetime64, check for date-like column names
        for col in df.columns:
            if col not in dt_cols:
                col_lower = str(col).lower()
                if any(k in col_lower for k in ["date", "time", "timestamp", "year", "month"]):
                    if n_rows > 0:
                        sample = df[col].dropna().head(10)
                        if not sample.empty:
                            try:
                                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
                                if parsed.notna().sum() >= len(sample) * 0.8:
                                    dt_cols.append(col)
                            except Exception:
                                pass

        # Normalized column map
        normalized_cols = {str(col): ColumnNormalizer.normalize(str(col)) for col in df.columns}

        # Unique counts & candidate key detection
        unique_counts = {}
        candidate_keys = []

        if n_rows > 0:
            for col in df.columns:
                try:
                    u_count = int(df[col].nunique())
                    unique_counts[str(col)] = u_count

                    # Candidate key heuristic:
                    # 1. Zero nulls
                    # 2. Strict uniqueness ratio >= 0.95
                    # 3. Exclude index-like columns (row numbers, indices, Unnamed, sequential numeric counters)
                    # 4. Require valid business identifier or key name semantics
                    # 5. Exclude continuous floats
                    null_count = int(df[col].isnull().sum())
                    if null_count == 0 and n_rows >= 2:
                        ratio = u_count / n_rows
                        if ratio >= 0.95:
                            # Filter out index-like columns
                            if ColumnNormalizer.is_index_like(str(col), df[col]):
                                continue

                            col_norm = normalized_cols.get(str(col), "")
                            is_biz = ColumnNormalizer.is_business_identifier(str(col))
                            is_key_name = any(k in col_norm for k in ["id", "code", "key", "number", "num"])
                            is_float = pd.api.types.is_float_dtype(df[col])

                            if (is_biz or is_key_name) and not is_float:
                                candidate_keys.append(str(col))
                except Exception:
                    unique_counts[str(col)] = 0
        else:
            for col in df.columns:
                unique_counts[str(col)] = 0

        # Memory usage
        try:
            mem_mb = round(df.memory_usage(deep=True).sum() / (1024**2), 2)
        except Exception:
            mem_mb = 0.0

        # Resolved filename
        resolved_filename = None
        if filename:
            resolved_filename = os.path.basename(filename)
        elif hasattr(df, "attrs") and df.attrs.get("filename"):
            resolved_filename = os.path.basename(df.attrs["filename"])

        sheet_name = df.attrs.get("sheet_name") if hasattr(df, "attrs") else None

        return DatasetProfile(
            rows=n_rows,
            columns=n_cols,
            column_names=[str(c) for c in df.columns],
            data_types={str(k): str(v) for k, v in df.dtypes.items()},
            missing_values={str(k): int(v) for k, v in df.isnull().sum().items()},
            duplicate_rows=int(df.duplicated().sum()) if n_rows > 0 else 0,
            memory_usage_mb=mem_mb,
            numerical_columns=[str(c) for c in df.select_dtypes(include="number").columns.tolist()],
            categorical_columns=[str(c) for c in cat_cols],
            filename=resolved_filename,
            sheet_name=sheet_name,
            datetime_columns=[str(c) for c in dt_cols],
            normalized_columns=normalized_cols,
            unique_counts=unique_counts,
            candidate_keys=candidate_keys,
        )