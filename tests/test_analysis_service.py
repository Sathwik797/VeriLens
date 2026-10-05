import unittest
from unittest.mock import MagicMock
import pandas as pd

from app.executor.safe_executor import ExecutionResult, SafeExecutor
from app.models.analysis_plan import AnalysisPlan
from app.models.verification import AnalyticalClaim, Claim, VerificationResult
from app.services.analysis_service import (
    AnalysisPipelineResult,
    AnalysisService,
    AnalysisServiceError,
)
from app.services.claim_extractor_service import ClaimExtractorService
from app.services.narrator_service import NarratorError, NarratorService
from app.services.planner_service import PlannerService
from app.services.verification_service import VerificationService


class TestAnalysisService(unittest.TestCase):

    def setUp(self):
        # Sample dataset for testing
        self.df = pd.DataFrame([
            {"Region": "West", "Profit": 108418.45, "Sales": 725457.82},
            {"Region": "East", "Profit": 91522.78, "Sales": 678781.24},
            {"Region": "Central", "Profit": 39706.36, "Sales": 501239.89},
            {"Region": "South", "Profit": 46749.43, "Sales": 391721.90},
        ])
        self.question = "Which region has the highest profit?"

        self.sample_plan = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
            limit=1,
            description="Find highest profit region.",
        )

    def test_01_successful_complete_pipeline(self):
        """1. Successful complete pipeline execution."""
        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = self.sample_plan

        mock_narrator = MagicMock(spec=NarratorService)
        mock_narrator.generate.return_value = "The West region generated the highest profit with $108,418.45."

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        mock_claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="highest",
            group_by="Region",
        )
        mock_extractor.extract_claims.return_value = [mock_claim]

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        result = service.analyze_question(self.df, self.question)

        self.assertIsInstance(result, AnalysisPipelineResult)
        self.assertEqual(result.question, self.question)
        self.assertEqual(result.analysis_plan, self.sample_plan)
        self.assertEqual(len(result.claims), 1)
        self.assertEqual(len(result.verification_results), 1)
        self.assertEqual(result.verification_results[0].status, "verified")
        self.assertTrue(result.verification_results[0].matched)

    def test_02_pipeline_execution_order(self):
        """2. Verifies components are called in strict order: Planner -> Executor -> Narrator -> Extractor."""
        call_order = []

        mock_planner = MagicMock(spec=PlannerService)
        def plan_side_effect(*args, **kwargs):
            call_order.append("planner")
            return self.sample_plan
        mock_planner.generate_plan.side_effect = plan_side_effect

        mock_narrator = MagicMock(spec=NarratorService)
        def narrator_side_effect(*args, **kwargs):
            call_order.append("narrator")
            return "Explanation text."
        mock_narrator.generate.side_effect = narrator_side_effect

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        def extractor_side_effect(*args, **kwargs):
            call_order.append("extractor")
            return []
        mock_extractor.extract_claims.side_effect = extractor_side_effect

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        service.analyze_question(self.df, self.question)
        self.assertEqual(call_order, ["planner", "narrator", "extractor"])

    def test_03_multiple_claims_produce_multiple_verification_results(self):
        """3. Multiple claims each produce an independent VerificationResult."""
        plan_all = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
        )
        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = plan_all

        mock_narrator = MagicMock(spec=NarratorService)
        mock_narrator.generate.return_value = "West had highest profit ($108K) and East had $91K."

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        mock_extractor.extract_claims.return_value = [
            Claim(subject="West", metric="Profit", value=108418.45, comparison="highest", group_by="Region"),
            Claim(subject="East", metric="Profit", value=91522.78, comparison="equal", group_by="Region"),
        ]

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        result = service.analyze_question(self.df, self.question)
        self.assertEqual(len(result.claims), 2)
        self.assertEqual(len(result.verification_results), 2)
        self.assertTrue(all(vr.matched for vr in result.verification_results))

    def test_04_verified_claim_preserved(self):
        """4. Verified claim is accurately preserved in pipeline result."""
        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = self.sample_plan

        mock_narrator = MagicMock(spec=NarratorService)
        mock_narrator.generate.return_value = "West generated $108,418.45."

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        mock_extractor.extract_claims.return_value = [
            Claim(subject="West", metric="Profit", value=108418.45, comparison="highest", group_by="Region")
        ]

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        result = service.analyze_question(self.df, self.question)
        self.assertEqual(result.verification_results[0].status, "verified")
        self.assertTrue(result.verification_results[0].matched)

    def test_05_mismatching_claim_preserved(self):
        """5. Mismatching claim (hallucination/error in narration) is flagged as mismatch without alteration."""
        plan_all = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
        )
        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = plan_all

        mock_narrator = MagicMock(spec=NarratorService)
        # Narrator incorrectly asserts East is highest
        mock_narrator.generate.return_value = "East had the highest profit."

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        mock_extractor.extract_claims.return_value = [
            Claim(subject="East", metric="Profit", comparison="highest", group_by="Region")
        ]

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        result = service.analyze_question(self.df, self.question)
        self.assertEqual(result.verification_results[0].status, "mismatch")
        self.assertFalse(result.verification_results[0].matched)
        self.assertIn("East", result.claims[0].subject)

    def test_06_inconclusive_verification_preserved(self):
        """6. Inconclusive verification is preserved when evidence is insufficient."""
        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = self.sample_plan

        mock_narrator = MagicMock(spec=NarratorService)
        mock_narrator.generate.return_value = "Profit was highest in EMEA."

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        mock_extractor.extract_claims.return_value = [
            Claim(subject="EMEA", metric="Profit", comparison="highest", group_by="Region")
        ]

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        result = service.analyze_question(self.df, self.question)
        self.assertEqual(result.verification_results[0].status, "inconclusive")
        self.assertFalse(result.verification_results[0].matched)

    def test_07_empty_claim_list_handled_cleanly(self):
        """7. Empty claim list produces an empty verification_results list without failing."""
        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = self.sample_plan

        mock_narrator = MagicMock(spec=NarratorService)
        mock_narrator.generate.return_value = "Here is some general information with no specific claims."

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        mock_extractor.extract_claims.return_value = []

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        result = service.analyze_question(self.df, self.question)
        self.assertEqual(result.claims, [])
        self.assertEqual(result.verification_results, [])

    def test_08_component_errors_not_swallowed(self):
        """8. Errors from underlying components are propagated and not silently swallowed."""
        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = self.sample_plan

        mock_narrator = MagicMock(spec=NarratorService)
        mock_narrator.generate.side_effect = NarratorError("LLM quota exceeded.")

        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
        )

        with self.assertRaises(NarratorError):
            service.analyze_question(self.df, self.question)

    def test_09_real_deterministic_components_integration(self):
        """
        9. Integration-style test:
        Real DataFrame -> Real AnalysisPlan -> Real SafeExecutor -> Mock Narrator -> Mock ClaimExtractor -> Real VerificationService.
        Proves that real execution evidence flows into the real verifier accurately.
        """
        real_plan = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
        )

        mock_planner = MagicMock(spec=PlannerService)
        mock_planner.generate_plan.return_value = real_plan

        mock_narrator = MagicMock(spec=NarratorService)
        mock_narrator.generate.return_value = "The West region generated highest profit."

        mock_extractor = MagicMock(spec=ClaimExtractorService)
        extracted_claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="highest",
            group_by="Region",
        )
        mock_extractor.extract_claims.return_value = [extracted_claim]

        # Real SafeExecutor and real VerificationService are executed by AnalysisService
        service = AnalysisService(
            planner_service=mock_planner,
            narrator_service=mock_narrator,
            claim_extractor_service=mock_extractor,
        )

        result = service.analyze_question(self.df, "Which region has the highest profit?")

        # Check real SafeExecutor result reached the pipeline
        self.assertIsInstance(result.execution_result, ExecutionResult)
        evidence = result.evidence
        self.assertEqual(evidence[0]["Region"], "West")
        self.assertAlmostEqual(evidence[0]["Profit"], 108418.45, places=2)

        # Check real VerificationService evaluated against real evidence
        self.assertEqual(len(result.verification_results), 1)
        vr = result.verification_results[0]
        self.assertEqual(vr.status, "verified")
        self.assertTrue(vr.matched)
        self.assertEqual(vr.actual_value, 108418.45)
        self.assertEqual(vr.evidence, evidence)


if __name__ == "__main__":
    unittest.main()
