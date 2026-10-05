from typing import Any, Dict, List, Literal, Optional, Union
from pydantic import BaseModel, Field

ComparisonType = Literal[
    "highest",
    "lowest",
    "equal",
    "greater_than",
    "less_than",
    "greater_than_or_equal",
    "less_than_or_equal",
]

VerificationStatus = Literal[
    "verified",
    "mismatch",
    "inconclusive",
]


class AnalyticalClaim(BaseModel):
    """
    Structured representation of an analytical claim made in an AI response.
    Captures factual assertions before verification against ground truth.
    """
    subject: Optional[str] = Field(
        default=None,
        description="The subject entity of the claim (e.g. 'West', 'Technology')."
    )
    metric: str = Field(
        ...,
        description="The quantitative or categorical metric evaluated (e.g. 'Profit', 'Sales')."
    )
    value: Optional[Union[float, int]] = Field(
        default=None,
        description="The claimed numerical value (e.g. 108418.45)."
    )
    comparison: Optional[ComparisonType] = Field(
        default=None,
        description="Controlled comparison or ranking assertion."
    )
    group_by: Optional[str] = Field(
        default=None,
        description="The dimension/column along which the entity was evaluated (e.g. 'Region')."
    )


# Alias for concise import
Claim = AnalyticalClaim


class VerificationResult(BaseModel):
    """
    Structured outcome of verifying an analytical claim against deterministic ground truth.
    """
    status: VerificationStatus = Field(
        ...,
        description="Outcome status: 'verified', 'mismatch', or 'inconclusive'."
    )
    matched: bool = Field(
        ...,
        description="True if the claim matches dataset evidence, False otherwise."
    )
    expected_value: Optional[Union[float, int, str, bool]] = Field(
        default=None,
        description="The claimed or expected value."
    )
    actual_value: Optional[Union[float, int, str, bool]] = Field(
        default=None,
        description="The deterministic ground-truth value computed from the dataset."
    )
    evidence: Optional[Union[List[Dict[str, Any]], Dict[str, Any]]] = Field(
        default=None,
        description="Structured records/rows from ExecutionResult serving as empirical proof."
    )
    reason: Optional[str] = Field(
        default=None,
        description="Human-readable explanation of the verification check."
    )
