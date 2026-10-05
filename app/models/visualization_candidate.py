from typing import Any, Dict, List, Literal, Optional
from pydantic import BaseModel, Field

VisualizationType = Literal["bar", "pie", "donut", "line", "scatter", "histogram"]


class VisualizationCandidate(BaseModel):
    """
    Structured descriptor for an exploratory visualization candidate.
    Encapsulates chart type, target columns, aggregation logic, suitability score,
    and semantic rationale without storing raw dataframe contents.
    """
    candidate_id: str
    chart_type: VisualizationType
    primary_column: str
    secondary_column: Optional[str] = None
    aggregation: Optional[Literal["sum", "mean", "count"]] = None
    title: str
    description: str
    suitability_score: float = Field(ge=0.0, le=100.0)
    metadata: Dict[str, Any] = Field(default_factory=dict)
