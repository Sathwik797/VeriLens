import os
import unittest
from unittest.mock import MagicMock, patch

from app.models.verification import AnalyticalClaim
from app.services.claim_extractor_service import (
    ClaimExtractionError,
    ClaimExtractorConfigurationError,
    ClaimExtractorService,
)


class TestClaimExtractorService(unittest.TestCase):

    def setUp(self):
        self.narrator_text = "The West region generated the highest profit with a total of $108,418.45."

    def test_01_successful_single_claim_extraction(self):
        """Test: Extracts a single valid AnalyticalClaim from narrator text."""
        service = ClaimExtractorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """
        {
            "claims": [
                {
                    "subject": "West",
                    "metric": "Profit",
                    "value": 108418.45,
                    "comparison": "highest",
                    "group_by": "Region"
                }
            ]
        }
        """
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            claims = service.extract_claims(self.narrator_text)

            self.assertEqual(len(claims), 1)
            claim = claims[0]
            self.assertIsInstance(claim, AnalyticalClaim)
            self.assertEqual(claim.subject, "West")
            self.assertEqual(claim.metric, "Profit")
            self.assertEqual(claim.value, 108418.45)
            self.assertEqual(claim.comparison, "highest")
            self.assertEqual(claim.group_by, "Region")

    def test_02_successful_multiple_claims_extraction(self):
        """Test: Extracts multiple distinct analytical claims."""
        service = ClaimExtractorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """
        {
            "claims": [
                {
                    "subject": "West",
                    "metric": "Profit",
                    "value": 108418.45,
                    "comparison": "highest",
                    "group_by": "Region"
                },
                {
                    "subject": "East",
                    "metric": "Profit",
                    "value": 91522.78,
                    "comparison": "equal",
                    "group_by": "Region"
                }
            ]
        }
        """
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            claims = service.extract_claims("West had $108K profit while East had $91K.")

            self.assertEqual(len(claims), 2)
            self.assertEqual(claims[0].subject, "West")
            self.assertEqual(claims[1].subject, "East")
            self.assertEqual(claims[1].value, 91522.78)

    def test_03_missing_api_key_raises_configuration_error(self):
        """Test: Missing API key raises ClaimExtractorConfigurationError."""
        service = ClaimExtractorService(api_key=None)
        with patch.dict(os.environ, {}, clear=True):
            service.api_key = None
            with self.assertRaises(ClaimExtractorConfigurationError):
                service.extract_claims(self.narrator_text)

    def test_04_empty_narrator_response_raises_error(self):
        """Test: Empty or whitespace narrator explanation raises ClaimExtractionError."""
        service = ClaimExtractorService(api_key="mock-key")
        with self.assertRaises(ClaimExtractionError) as ctx:
            service.extract_claims("   ")
        self.assertIn("cannot be empty", str(ctx.exception))

    def test_05_gemini_api_failure_raises_error(self):
        """Test: Upstream Gemini API error is caught and wrapped in ClaimExtractionError."""
        service = ClaimExtractorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_model.generate_content.side_effect = RuntimeError("Quota exceeded")

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            with self.assertRaises(ClaimExtractionError) as ctx:
                service.extract_claims(self.narrator_text)
            self.assertIn("Error calling Gemini API", str(ctx.exception))

    def test_06_malformed_json_response_raises_error(self):
        """Test: Non-JSON response from Gemini raises ClaimExtractionError."""
        service = ClaimExtractorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "This is definitely not valid JSON."
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            with self.assertRaises(ClaimExtractionError) as ctx:
                service.extract_claims(self.narrator_text)
            self.assertIn("Failed to parse Gemini output as JSON", str(ctx.exception))

    def test_07_invalid_claim_structure_raises_error(self):
        """Test: Claim with invalid comparison or missing required metric raises ClaimExtractionError."""
        service = ClaimExtractorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        # Invalid comparison type 'top_dog' not in allowed Literal
        mock_response.text = """
        {
            "claims": [
                {
                    "subject": "West",
                    "metric": "Profit",
                    "comparison": "top_dog"
                }
            ]
        }
        """
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            with self.assertRaises(ClaimExtractionError) as ctx:
                service.extract_claims(self.narrator_text)
            self.assertIn("does not conform to AnalyticalClaim schema", str(ctx.exception))

    def test_08_prompt_contains_narrator_explanation(self):
        """Test: Sent prompt includes the exact narrator explanation and optional question."""
        service = ClaimExtractorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = '{"claims": []}'
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            service.extract_claims(
                explanation="Technology sales surged by 45 percent.",
                question="What happened to technology?"
            )

            sent_prompt = mock_model.generate_content.call_args[0][0]
            self.assertIn("Technology sales surged by 45 percent.", sent_prompt)
            self.assertIn("What happened to technology?", sent_prompt)

    def test_09_returns_analytical_claim_instances(self):
        """Test: All returned items in the list are strictly AnalyticalClaim instances."""
        service = ClaimExtractorService(api_key="mock-key")

        mock_model = MagicMock()
        mock_response = MagicMock()
        mock_response.text = """
        {
            "claims": [
                {
                    "subject": "Corporate",
                    "metric": "Sales",
                    "value": 500000.0,
                    "comparison": "greater_than"
                }
            ]
        }
        """
        mock_model.generate_content.return_value = mock_response

        with patch("google.generativeai.GenerativeModel", return_value=mock_model), \
             patch("google.generativeai.configure"):

            results = service.extract_claims("Corporate sales exceeded 500k.")
            self.assertTrue(all(isinstance(c, AnalyticalClaim) for c in results))


if __name__ == "__main__":
    unittest.main()
