import unittest
from unittest.mock import MagicMock
import pandas as pd
import gradio as gr

from app.executor.safe_executor import ExecutionResult
from app.formatters.verification_formatter import VerificationFormatter
from app.models.analysis_plan import AnalysisPlan
from app.models.trust_score import TrustScore
from app.models.verification import AnalyticalClaim, Claim, VerificationResult
from app.services.analysis_service import AnalysisPipelineResult, AnalysisService
from ui.app import analyze_dataset, demo, process_question
import plotly.graph_objects as go


class TestUI(unittest.TestCase):

    def test_01_ui_app_imports_and_creates_blocks(self):
        """1. Verifies ui.app exports a valid Gradio Blocks instance."""
        self.assertIsInstance(demo, gr.Blocks)
        self.assertEqual(demo.title, "VeriLens AI")

    def test_01b_analyze_dataset_none_file(self):
        """1b. Verifies analyze_dataset handles None gracefully."""
        res = analyze_dataset(None)
        self.assertEqual(res, (None, None, None, None, None))

    def test_01c_analyze_dataset_returns_valid_profile_and_plotly_figures(self):
        """1c. Verifies analyze_dataset callback returns profile HTML and 4 valid Plotly Figure instances."""
        mock_file = MagicMock()
        mock_file.name = "datasets/superstore/Sample - Superstore.csv"

        profile_html, sales_plot, profit_plot, category_plot, region_plot = analyze_dataset(mock_file)

        # Verify profile HTML
        self.assertIsInstance(profile_html, str)
        self.assertIn("Rows", profile_html)
        self.assertIn("Columns", profile_html)

        # Verify all 4 are valid Plotly Figures
        self.assertIsInstance(sales_plot, go.Figure)
        self.assertIsInstance(profit_plot, go.Figure)
        self.assertIsInstance(category_plot, go.Figure)
        self.assertIsInstance(region_plot, go.Figure)

        # Verify figures have traces
        self.assertGreater(len(sales_plot.data), 0)
        self.assertGreater(len(profit_plot.data), 0)
        self.assertGreater(len(category_plot.data), 0)
        self.assertGreater(len(region_plot.data), 0)

    def test_02_no_file_uploaded_returns_friendly_message(self):
        """2. Calling process_question with no file produces a clear message."""
        answer, trust_html, claims_html, evidence_df = process_question(None, "Which region has highest profit?")
        self.assertIn("Please upload a CSV dataset first", answer)
        self.assertEqual(trust_html, "")
        self.assertEqual(claims_html, "")
        self.assertTrue(evidence_df.empty)

    def test_03_empty_question_returns_friendly_message(self):
        """3. Calling process_question with empty question produces a clear message."""
        mock_file = MagicMock()
        mock_file.name = "fake.csv"
        answer, trust_html, claims_html, evidence_df = process_question(mock_file, "   ")
        self.assertIn("Please enter a question", answer)
        self.assertEqual(trust_html, "")
        self.assertEqual(claims_html, "")
        self.assertTrue(evidence_df.empty)

    def test_04_successful_question_processing_with_mock_service(self):
        """4. Valid file and question with mock service produces formatted answer, trust score, claims, and evidence."""
        mock_file = MagicMock()
        mock_file.name = "datasets/superstore/Sample - Superstore.csv"

        # Mock AnalysisService result
        plan = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
            limit=1,
        )
        exec_res = ExecutionResult(
            data=pd.DataFrame([{"Region": "West", "Profit": 108418.45}]),
            operation="groupby",
            plan=plan,
        )
        claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="highest",
            group_by="Region",
        )
        vr = VerificationResult(
            status="verified",
            matched=True,
            expected_value=108418.45,
            actual_value=108418.45,
            evidence=[{"Region": "West", "Profit": 108418.45}],
            reason="Verified successfully.",
        )

        mock_result = AnalysisPipelineResult(
            question="Which region has highest profit?",
            analysis_plan=plan,
            execution_result=exec_res,
            explanation="The West region has the highest profit with $108,418.45.",
            claims=[claim],
            verification_results=[vr],
        )

        mock_service = MagicMock(spec=AnalysisService)
        mock_service.analyze_question.return_value = mock_result

        answer, trust_html, claims_html, evidence_df = process_question(
            file=mock_file,
            question="Which region has highest profit?",
            analysis_service=mock_service,
        )

        self.assertIn("The West region has the highest profit", answer)
        self.assertIn("High trust", trust_html)
        self.assertIn("100", trust_html)
        self.assertIn("✅ Verified", claims_html)
        self.assertIn("West", claims_html)
        self.assertFalse(evidence_df.empty)
        self.assertEqual(evidence_df.iloc[0]["Region"], "West")

    def test_05_verification_formatter_badges(self):
        """5. VerificationFormatter produces correct badges for verified, mismatch, and inconclusive."""
        claim1 = Claim(subject="West", metric="Profit", comparison="highest")
        vr1 = VerificationResult(status="verified", matched=True, actual_value=100.0)

        claim2 = Claim(subject="East", metric="Profit", comparison="highest")
        vr2 = VerificationResult(status="mismatch", matched=False, actual_value=50.0, reason="Not highest")

        claim3 = Claim(subject="North", metric="Sales", comparison="equal")
        vr3 = VerificationResult(status="inconclusive", matched=False, reason="Missing data")

        html = VerificationFormatter.claims_breakdown([claim1, claim2, claim3], [vr1, vr2, vr3])
        self.assertIn("✅ Verified", html)
        self.assertIn("❌ Mismatch", html)
        self.assertIn("⚠️ Inconclusive", html)

    def test_06_trust_score_card_formatting(self):
        """6. VerificationFormatter correctly formats trust score card."""
        ts = TrustScore(
            score=85.5,
            status="Moderate trust",
            total_claims=2,
            verified_claims=1,
            mismatch_claims=0,
            inconclusive_claims=1,
            explanation="1 of 2 analytical claims were verified.",
        )
        html = VerificationFormatter.trust_score_card(ts)
        self.assertIn("85.5", html)
        self.assertIn("Moderate trust", html)
        self.assertIn("1 of 2 analytical claims were verified.", html)


if __name__ == "__main__":
    unittest.main()
