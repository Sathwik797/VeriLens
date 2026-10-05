from typing import Any, List, Literal, Optional, Union
from pydantic import BaseModel, Field


class FilterCondition(BaseModel):
    """Represents an optional filter condition applied prior to aggregation."""
    column: str
    operator: Literal["==", "!=", ">", ">=", "<", "<=", "in", "contains"] = "=="
    value: Any


class AnalysisPlan(BaseModel):
    """
    Structured data analysis plan produced by the LLM Planner.
    Decouples reasoning (LLM) from execution (Pandas).
    """
    operation: Literal["groupby", "aggregate", "filter", "sort", "value_counts"] = Field(
        ...,
        description="The primary analytical operation to execute."
    )
    group_by: Optional[Union[str, List[str]]] = Field(
        default=None,
        description="Column or list of columns to group by (for groupby operations)."
    )
    metric: Optional[str] = Field(
        default=None,
        description="Target numerical or categorical column for aggregation/analysis."
    )
    aggregation: Optional[Literal["sum", "mean", "median", "count", "min", "max", "std"]] = Field(
        default=None,
        description="Aggregation function to apply."
    )
    filters: Optional[List[FilterCondition]] = Field(
        default=None,
        description="Optional list of row filter criteria to apply before aggregation."
    )
    sort: Optional[Literal["ascending", "descending"]] = Field(
        default=None,
        description="Sort direction of the aggregated results."
    )
    limit: Optional[int] = Field(
        default=None,
        description="Maximum number of rows to return (e.g. 1 for top/highest)."
    )
    description: Optional[str] = Field(
        default=None,
        description="Brief human-readable summary of what this plan calculates."
    )
