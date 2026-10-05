from typing import List, Literal, Optional
from pydantic import BaseModel, Field

JoinCardinality = Literal[
    "ONE_TO_ONE",
    "ONE_TO_MANY",
    "MANY_TO_ONE",
    "MANY_TO_MANY",
    "UNKNOWN",
]

RecommendedJoinType = Literal[
    "INNER_JOIN",
    "LEFT_JOIN",
    "RIGHT_JOIN",
    "FULL_OUTER_JOIN",
    "NO_SAFE_JOIN",
]

ConfidenceLevel = Literal[
    "HIGH",
    "MODERATE",
    "LOW",
    "VERY LOW",
]


class CandidateJoinKey(BaseModel):
    """
    Deterministic evaluation of a candidate key pair between two datasets.
    """
    left_column: str
    right_column: str
    left_dtype: str
    right_dtype: str
    left_unique: bool
    right_unique: bool
    left_null_count: int
    right_null_count: int
    left_null_percentage: float
    right_null_percentage: float
    matched_rows: int
    unmatched_left_rows: int
    unmatched_right_rows: int
    coverage_percentage: float
    cardinality: JoinCardinality
    confidence: int = Field(ge=0, le=100, description="Deterministic confidence score from 0 to 100")
    confidence_level: ConfidenceLevel
    recommended_join: RecommendedJoinType
    parent_dataset: Optional[str] = None
    parent_key: Optional[str] = None
    child_dataset: Optional[str] = None
    child_key: Optional[str] = None
    safe_to_execute: bool
    evidence: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)


class JoinRecommendation(BaseModel):
    """
    Deterministic, evidence-based recommendation for a safe join relationship
    between two datasets without executing any merge.
    """
    dataset_a: str
    dataset_b: str
    left_column: str
    right_column: str
    relationship_type: str = "RELATED"
    cardinality: JoinCardinality
    confidence: int = Field(ge=0, le=100, description="Deterministic confidence score from 0 to 100")
    confidence_level: ConfidenceLevel
    left_unique: bool
    right_unique: bool
    left_null_percentage: float
    right_null_percentage: float
    matched_rows: int
    unmatched_left_rows: int
    unmatched_right_rows: int
    coverage_percentage: float
    recommended_join: RecommendedJoinType
    parent_dataset: Optional[str] = None
    parent_key: Optional[str] = None
    child_dataset: Optional[str] = None
    child_key: Optional[str] = None
    safe_to_execute: bool
    evidence: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    explanation: str = ""
    all_candidates: List[CandidateJoinKey] = Field(default_factory=list)
