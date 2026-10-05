import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from app.executor.safe_executor import ExecutionResult, SafeExecutor
from app.models.analysis_plan import AnalysisPlan
from app.services.narrator_service import (
    NarratorConfigurationError,
    NarratorError,
    NarratorService,
)


class TestNarratorService(unittest.TestCase):

    def setUp(self):
        self.question = "Which region has the highest profit?"
        self.plan = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
            limit=1,
            description="Calculate total profit by region and pick the highest.",
        )
        self.df = pd.DataFrame([{"Region": "West", "Profit": 108418.4489}])
        self.execution_result = SafeExecutor.execute(self.df, self.plan)

    def test_01_successful_narration(self):
        """Test: Successful narration generation with mocked Gemini."""
        service = NarratorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "The West region generated the highest profit at $108,418.45."
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            explanation = service.generate(
                question=self.question,
                analysis_plan=self.plan,
                execution_result=self.execution_result,
            )

            self.assertIsInstance(explanation, str)
            self.assertEqual(explanation, "The West region generated the highest profit at $108,418.45.")

    def test_02_missing_api_key_configuration(self):
        """Test: Missing API key raises NarratorConfigurationError."""
        service = NarratorService(api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            service.api_key = None
            with self.assertRaises(NarratorConfigurationError):
                service.generate(
                    question=self.question,
                    analysis_plan=self.plan,
                    execution_result=self.execution_result,
                )

    def test_03_empty_execution_evidence(self):
        """Test: Empty execution evidence raises NarratorError."""
        service = NarratorService(api_key="mock-key")
        empty_res = ExecutionResult(data=pd.DataFrame(), operation="groupby", plan=self.plan)

        with self.assertRaises(NarratorError) as ctx:
            service.generate(
                question=self.question,
                analysis_plan=self.plan,
                execution_result=empty_res,
            )
        self.assertIn("no evidence records", str(ctx.exception))

    def test_04_gemini_api_failure(self):
        """Test: Gemini API call failure raises NarratorError."""
        service = NarratorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = Exception("API connection timeout")

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            with self.assertRaises(NarratorError) as ctx:
                service.generate(
                    question=self.question,
                    analysis_plan=self.plan,
                    execution_result=self.execution_result,
                )
            self.assertIn("Error calling Gemini API for narration", str(ctx.exception))

    def test_05_prompt_contains_question_and_evidence(self):
        """Test: Generated prompt explicitly includes the user question and evidence."""
        service = NarratorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Answer text."
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            service.generate(
                question=self.question,
                analysis_plan=self.plan,
                execution_result=self.execution_result,
            )

            # Inspect argument passed to generate_content
            self.assertTrue(mock_model.generate_content.called)
            sent_prompt = mock_model.generate_content.call_args[0][0]

            self.assertIn("Which region has the highest profit?", sent_prompt)
            self.assertIn("West", sent_prompt)
            self.assertIn("108418.4489", sent_prompt)
            self.assertIn("Profit", sent_prompt)

    def test_06_returns_string_explanation(self):
        """Test: Ensures the return value is a non-empty string."""
        service = NarratorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Based on the evidence, West had the top profit."
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            result = service.generate(
                question=self.question,
                analysis_plan=self.plan,
                execution_result=self.execution_result,
            )

            self.assertIsInstance(result, str)
            self.assertGreater(len(result), 0)


if __name__ == "__main__":
    unittest.main()
