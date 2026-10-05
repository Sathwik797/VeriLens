from typing import List, Literal, Optional
from pydantic import BaseModel, Field

from app.models.join_recommendation import JoinCardinality, RecommendedJoinType

JoinExecutionStatus = Literal[
    "SUCCESS",
    "SUCCESS_WITH_WARNINGS",
    "BLOCKED",
]


class JoinExecutionResult(BaseModel):
    """
    Deterministic, auditable outcome of a Safe Join Execution.
    Tracks provenance, row counts, matching statistics, validation metrics,
    and warnings without embedding the raw DataFrame directly.
    """
    status: JoinExecutionStatus
    safe_to_execute: bool
    left_dataset: str
    right_dataset: str
    left_column: str
    right_column: str
    join_type: RecommendedJoinType
    cardinality: JoinCardinality
    confidence: int = Field(ge=0, le=100)
    rows_before_left: int
    rows_before_right: int
    rows_after: int
    columns_before: int
    columns_after: int
    matched_rows: int
    unmatched_left_rows: int
    unmatched_right_rows: int
    row_multiplication_factor: float
    duplicate_expansion_detected: bool
    derived_dataset_name: str
    warnings: List[str] = Field(default_factory=list)
    evidence: List[str] = Field(default_factory=list)
    error_reason: Optional[str] = None
    execution_timestamp: str = ""
