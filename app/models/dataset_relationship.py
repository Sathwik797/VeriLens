from typing import List, Literal, Optional
from pydantic import BaseModel, Field


RelationshipType = Literal["COMPATIBLE", "RELATED", "UNRELATED"]


class DatasetRelationship(BaseModel):
    """
    Structured outcome of deterministic dataset compatibility analysis.
    Classifies relationship between dataset_a and dataset_b into
    COMPATIBLE, RELATED, or UNRELATED with evidence and confidence.
    """
    dataset_a: str
    dataset_b: str
    classification: RelationshipType
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score from 0.0 to 1.0")
    shared_columns: List[str] = Field(default_factory=list, description="Columns matching in schema/name")
    candidate_keys: List[str] = Field(default_factory=list, description="Potential foreign/primary key linkage")
    evidence: List[str] = Field(default_factory=list, description="Deterministic factual signals observed")
    recommendation: str = Field(default="", description="Prescriptive guidance for user")
