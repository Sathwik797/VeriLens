from pydantic import BaseModel
from typing import Dict, List


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