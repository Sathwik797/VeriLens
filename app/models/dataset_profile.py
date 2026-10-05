from typing import Dict, List, Optional
from pydantic import BaseModel, Field


class DatasetProfile(BaseModel):
    rows: int
    columns: int

    column_names: List[str]

    data_types: Dict[str, str]

    missing_values: Dict[str, int]

    duplicate_rows: int

    memory_usage_mb: float

    numerical_columns: List[str]

    categorical_columns: List[str]

    # Enhanced multi-dataset workspace metadata (optional, backward-compatible)
    filename: Optional[str] = None
    sheet_name: Optional[str] = None
    datetime_columns: List[str] = Field(default_factory=list)
    normalized_columns: Dict[str, str] = Field(default_factory=dict)
    unique_counts: Dict[str, int] = Field(default_factory=dict)
    candidate_keys: List[str] = Field(default_factory=list)