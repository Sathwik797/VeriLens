from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import pandas as pd

from app.models.analysis_plan import AnalysisPlan, FilterCondition


class ExecutorError(Exception):
    """Base exception for executor failures."""
    pass


class ExecutorColumnError(ExecutorError):
    """Raised when a referenced column does not exist in the DataFrame."""
    pass


class ExecutorOperationError(ExecutorError):
    """Raised when an operation or aggregation is unsupported or invalid."""
    pass


class ExecutorFilterError(ExecutorError):
    """Raised when a filter condition cannot be applied."""
    pass


@dataclass
class ExecutionResult:
    """
    Encapsulates the deterministic computation output.
    Provides direct access to raw Pandas results and structured records for verification.
    """
    data: Union[pd.DataFrame, pd.Series, Any]
    operation: str
    plan: AnalysisPlan

    def to_records(self) -> List[Dict[str, Any]]:
        """Converts tabular result into a list of row dictionaries."""
        if isinstance(self.data, pd.DataFrame):
            return self.data.to_dict(orient="records")
        elif isinstance(self.data, pd.Series):
            return self.data.reset_index().to_dict(orient="records")
        return [{"value": self.data}]

    @property
    def scalar_value(self) -> Optional[Any]:
        """Returns the single scalar value if result is a single-value aggregation."""
        if isinstance(self.data, (int, float, str, bool)):
            return self.data
        if isinstance(self.data, pd.Series) and len(self.data) == 1:
            return self.data.iloc[0]
        if isinstance(self.data, pd.DataFrame) and self.data.shape == (1, 1):
            return self.data.iloc[0, 0]
        return None


class SafeExecutor:
    """
    Deterministic executor for AnalysisPlan objects.
    Executes analytical queries via explicit Pandas operations without string code execution.
    """

    ALLOWED_AGGREGATIONS = {
        "sum",
        "mean",
        "median",
        "count",
        "min",
        "max",
        "std",
    }

    ALLOWED_OPERATIONS = {
        "groupby",
        "aggregate",
        "filter",
        "sort",
        "value_counts",
    }

    @classmethod
    def execute(cls, df: pd.DataFrame, plan: AnalysisPlan) -> ExecutionResult:
        """
        Executes an AnalysisPlan against the provided DataFrame.

        Args:
            df: The pandas DataFrame to query.
            plan: Validated AnalysisPlan model.

        Returns:
            ExecutionResult containing computed Pandas data and records.
        """
        if not isinstance(df, pd.DataFrame):
            raise ExecutorError(f"Expected pandas DataFrame, got {type(df).__name__}")

        if not isinstance(plan, AnalysisPlan):
            raise ExecutorError(f"Expected AnalysisPlan, got {type(plan).__name__}")

        if plan.operation not in cls.ALLOWED_OPERATIONS:
            raise ExecutorOperationError(f"Unsupported operation: {plan.operation}")

        # 1. Validate column existence
        cls._validate_columns(df, plan)

        # 2. Apply filters (if any)
        working_df = df
        if plan.filters:
            working_df = cls._apply_filters(working_df, plan.filters)

        # 3. Route to dedicated operation handler
        if plan.operation == "groupby":
            result_data = cls._execute_groupby(working_df, plan)
        elif plan.operation == "aggregate":
            result_data = cls._execute_aggregate(working_df, plan)
        elif plan.operation == "filter":
            result_data = cls._execute_filter(working_df, plan)
        elif plan.operation == "sort":
            result_data = cls._execute_sort(working_df, plan)
        elif plan.operation == "value_counts":
            result_data = cls._execute_value_counts(working_df, plan)
        else:
            raise ExecutorOperationError(f"Unhandled operation: {plan.operation}")

        return ExecutionResult(data=result_data, operation=plan.operation, plan=plan)

    @classmethod
    def _validate_columns(cls, df: pd.DataFrame, plan: AnalysisPlan) -> None:
        """Verifies that all columns referenced in the plan exist in the DataFrame."""
        available_cols = set(df.columns)

        def check_col(col: Optional[str], context: str) -> None:
            if col and col not in available_cols:
                raise ExecutorColumnError(
                    f"Column '{col}' referenced in {context} not found in dataset. "
                    f"Available columns: {sorted(list(available_cols))}"
                )

        # Check group_by columns
        if plan.group_by:
            if isinstance(plan.group_by, list):
                for col in plan.group_by:
                    check_col(col, "group_by")
            else:
                check_col(plan.group_by, "group_by")

        # Check metric column
        if plan.metric:
            check_col(plan.metric, "metric")

        # Check filter columns
        if plan.filters:
            for f in plan.filters:
                check_col(f.column, "filter")

    @classmethod
    def _apply_filters(cls, df: pd.DataFrame, filters: List[FilterCondition]) -> pd.DataFrame:
        """Applies filter conditions using controlled boolean masking."""
        filtered_df = df
        for cond in filters:
            col = cond.column
            op = cond.operator
            val = cond.value

            if col not in filtered_df.columns:
                raise ExecutorColumnError(f"Filter column '{col}' not found.")

            series = filtered_df[col]

            if op == "==":
                mask = (series == val)
            elif op == "!=":
                mask = (series != val)
            elif op == ">":
                mask = (series > val)
            elif op == ">=":
                mask = (series >= val)
            elif op == "<":
                mask = (series < val)
            elif op == "<=":
                mask = (series <= val)
            elif op == "in":
                val_list = val if isinstance(val, (list, tuple, set)) else [val]
                mask = series.isin(val_list)
            elif op == "contains":
                mask = series.astype(str).str.contains(str(val), case=False, na=False)
            else:
                raise ExecutorFilterError(f"Unsupported filter operator: '{op}'")

            filtered_df = filtered_df[mask]

        return filtered_df

    @classmethod
    def _execute_groupby(cls, df: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
        """Executes a controlled groupby aggregation."""
        if not plan.group_by:
            raise ExecutorOperationError("Operation 'groupby' requires 'group_by' field.")
        if not plan.metric:
            raise ExecutorOperationError("Operation 'groupby' requires 'metric' field.")

        agg_func = plan.aggregation or "sum"
        if agg_func not in cls.ALLOWED_AGGREGATIONS:
            raise ExecutorOperationError(f"Unsupported aggregation: '{agg_func}'")

        group_cols = [plan.group_by] if isinstance(plan.group_by, str) else plan.group_by

        # Perform groupby aggregation deterministically
        grouped = df.groupby(group_cols, as_index=False)[plan.metric].agg(agg_func)

        # Apply sorting if specified
        if plan.sort:
            ascending = (plan.sort == "ascending")
            grouped = grouped.sort_values(by=plan.metric, ascending=ascending).reset_index(drop=True)

        # Apply limit if specified
        if plan.limit is not None and plan.limit > 0:
            grouped = grouped.head(plan.limit)

        return grouped

    @classmethod
    def _execute_aggregate(cls, df: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
        """Executes a scalar dataset-level aggregation."""
        if not plan.metric:
            raise ExecutorOperationError("Operation 'aggregate' requires 'metric' field.")

        agg_func = plan.aggregation or "sum"
        if agg_func not in cls.ALLOWED_AGGREGATIONS:
            raise ExecutorOperationError(f"Unsupported aggregation: '{agg_func}'")

        series = df[plan.metric]

        if agg_func == "sum":
            val = series.sum()
        elif agg_func == "mean":
            val = series.mean()
        elif agg_func == "median":
            val = series.median()
        elif agg_func == "count":
            val = series.count()
        elif agg_func == "min":
            val = series.min()
        elif agg_func == "max":
            val = series.max()
        elif agg_func == "std":
            val = series.std()
        else:
            raise ExecutorOperationError(f"Unsupported aggregation: '{agg_func}'")

        return pd.DataFrame([{plan.metric: val}])

    @classmethod
    def _execute_filter(cls, df: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
        """Returns filtered DataFrame, applying optional sort and limit."""
        result = df
        if plan.sort and plan.metric:
            ascending = (plan.sort == "ascending")
            result = result.sort_values(by=plan.metric, ascending=ascending).reset_index(drop=True)

        if plan.limit is not None and plan.limit > 0:
            result = result.head(plan.limit)

        return result

    @classmethod
    def _execute_sort(cls, df: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
        """Sorts the DataFrame by the requested metric column."""
        if not plan.metric:
            raise ExecutorOperationError("Operation 'sort' requires 'metric' column.")

        ascending = (plan.sort == "ascending") if plan.sort else True
        sorted_df = df.sort_values(by=plan.metric, ascending=ascending).reset_index(drop=True)

        if plan.limit is not None and plan.limit > 0:
            sorted_df = sorted_df.head(plan.limit)

        return sorted_df

    @classmethod
    def _execute_value_counts(cls, df: pd.DataFrame, plan: AnalysisPlan) -> pd.DataFrame:
        """Computes value counts for the target column."""
        target_col = plan.metric
        if not target_col and plan.group_by:
            target_col = plan.group_by[0] if isinstance(plan.group_by, list) else plan.group_by

        if not target_col:
            raise ExecutorOperationError("Operation 'value_counts' requires 'metric' or 'group_by' column.")

        counts = df[target_col].value_counts().reset_index()
        counts.columns = [target_col, "count"]

        if plan.sort:
            ascending = (plan.sort == "ascending")
            counts = counts.sort_values(by="count", ascending=ascending).reset_index(drop=True)

        if plan.limit is not None and plan.limit > 0:
            counts = counts.head(plan.limit)

        return counts
