import json
import os
from typing import Any, Dict, Optional, Union

from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import ValidationError

from app.models.analysis_plan import AnalysisPlan
from app.models.dataset_profile import DatasetProfile

load_dotenv()


class PlannerConfigurationError(Exception):
    """Raised when required planner configuration (such as GEMINI_API_KEY) is missing."""
    pass


class PlannerParsingError(Exception):
    """Raised when the LLM response cannot be parsed as valid JSON."""
    pass


class PlannerValidationError(Exception):
    """Raised when the parsed JSON does not conform to the AnalysisPlan Pydantic model."""
    pass


PLANNER_SYSTEM_INSTRUCTION = """You are the planning component of a data-analysis system.
Your job is to translate the user's natural-language question into a structured analysis plan.
Do not calculate the answer.
Do not invent values.
Do not write Python code.
Use only columns provided in the dataset schema.
Return ONLY valid JSON matching the AnalysisPlan schema.

Allowed operations:
- "groupby": For questions aggregating metrics across one or more categorical columns (e.g., "Which region has the highest profit?"). Requires group_by, metric, aggregation.
- "aggregate": For dataset-wide scalar calculations without grouping (e.g., "What is the total sales?"). Requires metric, aggregation.
- "filter": For querying rows matching specific conditions.
- "sort": For sorting dataset rows by a column.
- "value_counts": For counting occurrences of distinct values in a categorical column.

Allowed aggregation functions: "sum", "mean", "median", "count", "min", "max", "std".

JSON output structure:
{
  "operation": "groupby" | "aggregate" | "filter" | "sort" | "value_counts",
  "group_by": "ColumnName" or ["Col1", "Col2"] or null,
  "metric": "ColumnName" or null,
  "aggregation": "sum" | "mean" | "median" | "count" | "min" | "max" | "std" or null,
  "filters": [
    {
      "column": "ColumnName",
      "operator": "==" | "!=" | ">" | ">=" | "<" | "<=" | "in" | "contains",
      "value": "comparison_value"
    }
  ] or null,
  "sort": "ascending" | "descending" or null,
  "limit": integer or null,
  "description": "Brief summary of what this plan computes"
}
"""


class PlannerService:
    """
    Translates natural-language analytical questions into validated AnalysisPlan models
    using Google Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def _ensure_configured(self) -> None:
        """Validates that the Gemini API key is available and initializes the SDK."""
        if not self.api_key:
            raise PlannerConfigurationError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY "
                "in your .env file or environment variables."
            )
        genai.configure(api_key=self.api_key)

    @staticmethod
    def _extract_schema_context(schema: Union[DatasetProfile, Dict[str, Any]]) -> Dict[str, Any]:
        """Normalizes DatasetProfile or raw dictionary into a structured schema context."""
        if isinstance(schema, DatasetProfile):
            return {
                "columns": schema.column_names,
                "numerical_columns": schema.numerical_columns,
                "categorical_columns": schema.categorical_columns,
                "data_types": schema.data_types,
            }
        elif isinstance(schema, dict):
            return {
                "columns": schema.get("columns", schema.get("column_names", [])),
                "numerical_columns": schema.get("numerical_columns", []),
                "categorical_columns": schema.get("categorical_columns", []),
                "data_types": schema.get("data_types", {}),
            }
        else:
            raise ValueError(f"Unsupported schema type: {type(schema)}. Expected DatasetProfile or dict.")

    def _build_prompt(self, question: str, schema_context: Dict[str, Any]) -> str:
        """Constructs the prompt containing the user question and the dataset schema."""
        return f"""User Question:
\"{question}\"

Dataset Schema:
- Columns: {schema_context.get('columns', [])}
- Numerical Columns: {schema_context.get('numerical_columns', [])}
- Categorical Columns: {schema_context.get('categorical_columns', [])}
- Data Types: {schema_context.get('data_types', {})}

Generate the structured JSON analysis plan for this question adhering strictly to the columns above.
Return ONLY valid JSON matching the AnalysisPlan schema. Do not enclose in backticks or markdown if possible.
"""

    @staticmethod
    def _clean_json_text(text: str) -> str:
        """Strips markdown code fences from the raw LLM response text."""
        cleaned = text.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        elif cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    def generate_plan(
        self,
        question: str,
        schema: Union[DatasetProfile, Dict[str, Any]]
    ) -> AnalysisPlan:
        """
        Translates a natural language question into a validated AnalysisPlan.

        Args:
            question: The user's natural language question.
            schema: The DatasetProfile or schema dictionary with column context.

        Returns:
            Validated AnalysisPlan instance.

        Raises:
            PlannerConfigurationError: When GEMINI_API_KEY is missing.
            PlannerParsingError: When LLM response is not valid JSON.
            PlannerValidationError: When JSON does not match the AnalysisPlan schema.
        """
        if not question or not question.strip():
            raise ValueError("Question cannot be empty.")

        self._ensure_configured()
        schema_context = self._extract_schema_context(schema)
        prompt = self._build_prompt(question.strip(), schema_context)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=PLANNER_SYSTEM_INSTRUCTION,
                generation_config={"response_mime_type": "application/json"}
            )
            response = model.generate_content(prompt)
            raw_text = response.text
        except Exception as e:
            # Preserve original error context
            raise RuntimeError(f"Error calling Gemini API: {e}") from e

        cleaned_json = self._clean_json_text(raw_text)

        try:
            plan_dict = json.loads(cleaned_json)
        except json.JSONDecodeError as err:
            raise PlannerParsingError(
                f"Failed to parse Gemini output as JSON: {err}. Raw output was: {raw_text}"
            ) from err

        try:
            validated_plan = AnalysisPlan.model_validate(plan_dict)
        except ValidationError as err:
            raise PlannerValidationError(
                f"Generated plan does not conform to AnalysisPlan schema: {err}."
            ) from err

        return validated_plan
