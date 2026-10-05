from typing import Literal
from pydantic import BaseModel, Field

TrustStatus = Literal[
    "High trust",
    "Moderate trust",
    "Low trust",
    "Very low trust",
]


class TrustScore(BaseModel):
    """
    Deterministic explainable trust score summarizing verification results.
    """
    score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description="Overall trust score between 0.0 and 100.0."
    )
    status: TrustStatus = Field(
        ...,
        description="Categorical trust level: 'High trust', 'Moderate trust', 'Low trust', or 'Very low trust'."
    )
    total_claims: int = Field(
        ...,
        ge=0,
        description="Total number of evaluated analytical claims."
    )
    verified_claims: int = Field(
        ...,
        ge=0,
        description="Number of verified claims matching dataset ground truth."
    )
    mismatch_claims: int = Field(
        ...,
        ge=0,
        description="Number of claims contradicting dataset evidence."
    )
    inconclusive_claims: int = Field(
        ...,
        ge=0,
        description="Number of claims that could not be conclusively evaluated."
    )
    explanation: str = Field(
        ...,
        description="Deterministic human-readable explanation of the trust score."
    )
