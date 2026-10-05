import os
import unittest
from unittest.mock import MagicMock, patch

from pydantic import ValidationError

from app.models.analysis_plan import AnalysisPlan
from app.models.dataset_profile import DatasetProfile
from app.services.planner_service import (
    PlannerConfigurationError,
    PlannerParsingError,
    PlannerService,
    PlannerValidationError,
)


class TestPlannerService(unittest.TestCase):

    def setUp(self):
        # Sample schema based on Superstore dataset
        self.sample_schema = {
            "columns": ["Row ID", "Order ID", "Region", "Category", "Sales", "Profit"],
            "numerical_columns": ["Sales", "Profit"],
            "categorical_columns": ["Region", "Category"],
            "data_types": {
                "Row ID": "int64",
                "Order ID": "object",
                "Region": "object",
                "Category": "object",
                "Sales": "float64",
                "Profit": "float64",
            },
        }

    def test_missing_api_key_raises_configuration_error(self):
        """Service must raise PlannerConfigurationError if no API key is provided or found."""
        service = PlannerService(api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            service.api_key = None
            with self.assertRaises(PlannerConfigurationError):
                service.generate_plan("Which region has the highest profit?", self.sample_schema)

    def test_invalid_plan_rejected_by_pydantic(self):
        """Pydantic model must reject an invalid plan (e.g. unknown operation or illegal aggregation)."""
        invalid_plan_dict = {
            "operation": "unsupported_operation",
            "group_by": "Region",
            "metric": "Profit",
            "aggregation": "invalid_agg",
        }
        with self.assertRaises(ValidationError):
            AnalysisPlan.model_validate(invalid_plan_dict)

    def test_mocked_gemini_generates_valid_plan(self):
        """
        Simulate Gemini returning a structured plan for:
        'Which region has the highest profit?'
        Verify PlannerService parses and validates it into an AnalysisPlan.
        """
        service = PlannerService(api_key="dummy-test-key")

        mock_response_json = """
        {
            "operation": "groupby",
            "group_by": "Region",
            "metric": "Profit",
            "aggregation": "sum",
            "sort": "descending",
            "limit": 1,
            "description": "Calculate total profit by region and pick the highest one."
        }
        """

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = mock_response_json
        mock_model_instance.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance), \
             patch("google.generativeai.configure"):

            plan = service.generate_plan(
                question="Which region has the highest profit?",
                schema=self.sample_schema,
            )

            self.assertIsInstance(plan, AnalysisPlan)
            self.assertEqual(plan.operation, "groupby")
            self.assertEqual(plan.group_by, "Region")
            self.assertEqual(plan.metric, "Profit")
            self.assertEqual(plan.aggregation, "sum")
            self.assertEqual(plan.sort, "descending")
            self.assertEqual(plan.limit, 1)

    def test_malformed_json_raises_planner_parsing_error(self):
        """Service must raise PlannerParsingError when Gemini output is not valid JSON."""
        service = PlannerService(api_key="dummy-test-key")

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is plain English text, not JSON."
        mock_model_instance.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance), \
             patch("google.generativeai.configure"):

            with self.assertRaises(PlannerParsingError):
                service.generate_plan(
                    question="Which region has highest profit?",
                    schema=self.sample_schema,
                )

    def test_schema_mismatch_raises_planner_validation_error(self):
        """Service must raise PlannerValidationError when JSON doesn't conform to AnalysisPlan schema."""
        service = PlannerService(api_key="dummy-test-key")

        # Missing required 'operation' field
        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"group_by": "Region", "metric": "Profit"}'
        mock_model_instance.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance), \
             patch("google.generativeai.configure"):

            with self.assertRaises(PlannerValidationError):
                service.generate_plan(
                    question="Which region has highest profit?",
                    schema=self.sample_schema,
                )

    def test_with_dataset_profile_instance(self):
        """Service should accept a DatasetProfile model instance directly."""
        profile = DatasetProfile(
            rows=100,
            columns=3,
            column_names=["Region", "Sales", "Profit"],
            data_types={"Region": "object", "Sales": "float64", "Profit": "float64"},
            missing_values={"Region": 0, "Sales": 0, "Profit": 0},
            duplicate_rows=0,
            memory_usage_mb=0.1,
            numerical_columns=["Sales", "Profit"],
            categorical_columns=["Region"],
        )

        service = PlannerService(api_key="dummy-test-key")

        mock_model_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"operation": "aggregate", "metric": "Sales", "aggregation": "sum"}'
        mock_model_instance.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model_instance), \
             patch("google.generativeai.configure"):

            plan = service.generate_plan("What is total sales?", profile)
            self.assertEqual(plan.operation, "aggregate")
            self.assertEqual(plan.metric, "Sales")
            self.assertEqual(plan.aggregation, "sum")


if __name__ == "__main__":
    unittest.main()
