import json
import os
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
import google.generativeai as genai

from app.executor.safe_executor import ExecutionResult
from app.models.analysis_plan import AnalysisPlan

load_dotenv()


class NarratorConfigurationError(Exception):
    """Raised when required narrator configuration (such as GEMINI_API_KEY) is missing."""
    pass


class NarratorError(Exception):
    """Raised when the narration cannot be generated or input data is invalid."""
    pass


NARRATOR_SYSTEM_INSTRUCTION = """You are the narration component of an analytical system called VeriLens AI.
Your job is to translate ground-truth analytical computation results into a concise, accurate, natural-language response for the user.

Strict Rules:
1. Explain the result using ONLY the supplied ground-truth evidence records.
2. Never invent numbers, entities, rankings, or facts.
3. Never perform a new calculation or extrapolate numbers that are not represented in the evidence.
4. Never fabricate missing information.
5. Do not write Python code.
6. Do not mention internal implementation details (such as AnalysisPlan, Pandas, or SQL) unless explicitly asked.
7. Give a direct, concise, and helpful answer to the user's question.
8. If the evidence is insufficient to answer the question, explicitly state that the available evidence is insufficient.
9. Do not declare that your answer is 'verified' or 'guaranteed'.
"""


class NarratorService:
    """
    Translates ground-truth ExecutionResult data and user questions into
    natural-language analytical responses using Gemini.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None
    ):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.model_name = model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

    def _ensure_configured(self) -> None:
        """Validates that the Gemini API key is configured."""
        if not self.api_key:
            raise NarratorConfigurationError(
                "GEMINI_API_KEY is not configured. Please set GEMINI_API_KEY "
                "in your .env file or environment variables."
            )
        genai.configure(api_key=self.api_key)

    def _build_prompt(
        self,
        question: str,
        plan: AnalysisPlan,
        evidence: List[Dict[str, Any]],
    ) -> str:
        """Builds a strict narration prompt presenting the question, plan summary, and evidence."""
        plan_summary = {
            "operation": plan.operation,
            "metric": plan.metric,
            "group_by": plan.group_by,
            "aggregation": plan.aggregation,
            "limit": plan.limit,
            "description": plan.description,
        }
        filtered_plan = {k: v for k, v in plan_summary.items() if v is not None}

        return f"""User Question:
\"{question}\"

Analytical Operation Performed:
{json.dumps(filtered_plan, indent=2)}

Ground-Truth Evidence:
{json.dumps(evidence, indent=2, default=str)}

Provide a clear and concise natural-language response answering the user's question based strictly on the ground-truth evidence above.
"""

    def generate(
        self,
        question: str,
        analysis_plan: AnalysisPlan,
        execution_result: ExecutionResult,
    ) -> str:
        """
        Generates a natural-language analytical explanation of the execution result.

        Args:
            question: The user's original natural-language query.
            analysis_plan: The plan describing what was calculated.
            execution_result: Ground-truth result from SafeExecutor.

        Returns:
            Natural-language analytical explanation.

        Raises:
            NarratorConfigurationError: When GEMINI_API_KEY is missing.
            NarratorError: When inputs are invalid, evidence is empty, or API call fails.
        """
        if not question or not question.strip():
            raise NarratorError("Question cannot be empty.")
        if not isinstance(analysis_plan, AnalysisPlan):
            raise NarratorError(f"Expected AnalysisPlan, got {type(analysis_plan).__name__}.")
        if not isinstance(execution_result, ExecutionResult):
            raise NarratorError(f"Expected ExecutionResult, got {type(execution_result).__name__}.")

        records = execution_result.to_records()
        if not records:
            raise NarratorError("ExecutionResult contains no evidence records to narrate.")

        self._ensure_configured()
        prompt = self._build_prompt(question.strip(), analysis_plan, records)

        try:
            model = genai.GenerativeModel(
                model_name=self.model_name,
                system_instruction=NARRATOR_SYSTEM_INSTRUCTION,
            )
            response = model.generate_content(prompt)
            if not response.text or not response.text.strip():
                raise NarratorError("Received empty response from Gemini.")
            return response.text.strip()
        except NarratorError:
            raise
        except Exception as e:
            raise NarratorError(f"Error calling Gemini API for narration: {e}") from e
