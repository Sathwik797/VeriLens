import unittest
import pandas as pd

from app.executor.safe_executor import ExecutionResult, SafeExecutor
from app.models.analysis_plan import AnalysisPlan
from app.models.verification import AnalyticalClaim, Claim, VerificationResult
from app.services.verification_service import VerificationService


class TestVerificationService(unittest.TestCase):

    def setUp(self):
        # Sample regional profit evidence dataset
        self.evidence_df = pd.DataFrame([
            {"Region": "West", "Profit": 108418.4489},
            {"Region": "East", "Profit": 91522.7800},
            {"Region": "South", "Profit": 46749.4303},
            {"Region": "Central", "Profit": 39706.3625},
        ])

        self.plan = AnalysisPlan(
            operation="groupby",
            group_by="Region",
            metric="Profit",
            aggregation="sum",
            sort="descending",
        )
        self.execution_result = SafeExecutor.execute(self.evidence_df, self.plan)

    def test_01_highest_claim_correctly_verified(self):
        """1. Highest claim correctly verified."""
        claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="highest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)
        self.assertAlmostEqual(result.actual_value, 108418.4489, places=2)

    def test_02_highest_claim_correctly_rejected(self):
        """2. Highest claim correctly rejected (East is claimed highest, but West is highest)."""
        claim = Claim(
            subject="East",
            metric="Profit",
            value=91522.78,
            comparison="highest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "mismatch")
        self.assertFalse(result.matched)
        self.assertIn("does not have the highest", result.reason)

    def test_03_lowest_claim_correctly_verified(self):
        """3. Lowest claim correctly verified."""
        claim = Claim(
            subject="Central",
            metric="Profit",
            value=39706.36,
            comparison="lowest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)

    def test_04_lowest_claim_correctly_rejected(self):
        """4. Lowest claim correctly rejected (West claimed lowest)."""
        claim = Claim(
            subject="West",
            metric="Profit",
            comparison="lowest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "mismatch")
        self.assertFalse(result.matched)

    def test_05_equal_numerical_claim_verified_within_tolerance(self):
        """5. Equal numerical claim verified within tolerance."""
        claim = Claim(
            subject="East",
            metric="Profit",
            value=91522.78,
            comparison="equal",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)

    def test_06_equal_numerical_claim_rejected_outside_tolerance(self):
        """6. Equal numerical claim rejected outside tolerance."""
        claim = Claim(
            subject="East",
            metric="Profit",
            value=95000.00,  # Clearly different from 91522.78
            comparison="equal",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "mismatch")
        self.assertFalse(result.matched)

    def test_07_greater_than_claim(self):
        """7. Greater-than claim: West profit > 100,000."""
        claim_true = Claim(
            subject="West",
            metric="Profit",
            value=100000.0,
            comparison="greater_than",
            group_by="Region",
        )
        res_true = VerificationService.verify(claim_true, self.execution_result)
        self.assertEqual(res_true.status, "verified")
        self.assertTrue(res_true.matched)

        claim_false = Claim(
            subject="Central",
            metric="Profit",
            value=50000.0,
            comparison="greater_than",
            group_by="Region",
        )
        res_false = VerificationService.verify(claim_false, self.execution_result)
        self.assertEqual(res_false.status, "mismatch")
        self.assertFalse(res_false.matched)

    def test_08_less_than_claim(self):
        """8. Less-than claim: South profit < 50,000."""
        claim = Claim(
            subject="South",
            metric="Profit",
            value=50000.0,
            comparison="less_than",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)

    def test_09_missing_subject_returns_inconclusive(self):
        """9. Missing subject returns inconclusive."""
        claim = Claim(
            subject="NorthPole",  # Subject not in dataset
            metric="Profit",
            comparison="highest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "inconclusive")
        self.assertFalse(result.matched)
        self.assertIn("was not found", result.reason)

    def test_10_missing_metric_or_evidence_returns_inconclusive(self):
        """10. Missing metric returns inconclusive."""
        claim = Claim(
            subject="West",
            metric="Discount",  # Column not in evidence
            comparison="highest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "inconclusive")
        self.assertFalse(result.matched)
        self.assertIn("not found in evidence", result.reason)

    def test_11_floating_point_108418_4489_matches_claimed_108418_45(self):
        """11. Actual floating-point 108418.4489 matches claimed value 108418.45."""
        claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="equal",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)
        self.assertEqual(result.expected_value, 108418.45)
        self.assertEqual(result.actual_value, 108418.4489)

    def test_12_evidence_is_included_in_verification_result(self):
        """12. Evidence is included in VerificationResult."""
        claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="highest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertIsNotNone(result.evidence)
        self.assertIsInstance(result.evidence, list)
        self.assertEqual(len(result.evidence), 4)
        self.assertEqual(result.evidence[0]["Region"], "West")

    def test_13_lowercase_metric_resolves_to_titlecase_evidence(self):
        """13. 'Profit' in evidence + lowercase 'profit' claim metric resolves and verifies."""
        claim = Claim(
            subject="West",
            metric="profit",  # Lowercase metric name
            value=108418.45,
            comparison="highest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)
        self.assertEqual(result.actual_value, 108418.4489)

    def test_14_lowercase_group_by_resolves_to_titlecase_evidence(self):
        """14. 'Region' in evidence + lowercase 'region' claim group_by resolves and verifies."""
        claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="highest",
            group_by="region",  # Lowercase group_by
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)

    def test_15_unrelated_column_remains_inconclusive(self):
        """15. Completely unrelated column name ('profitability' vs 'Profit') remains INCONCLUSIVE."""
        claim = Claim(
            subject="West",
            metric="profitability",  # Unrelated/fuzzy name
            comparison="highest",
            group_by="Region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "inconclusive")
        self.assertFalse(result.matched)
        self.assertIn("not found in evidence", result.reason)

    def test_16_mixed_case_claim_fields_resolve_correctly(self):
        """16. Mixed-case claim fields ('pRoFiT', 'rEgIoN') resolve and verify correctly."""
        claim = Claim(
            subject="West",
            metric="pRoFiT",  # Mixed case
            value=108418.45,
            comparison="highest",
            group_by="rEgIoN",  # Mixed case
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)

    def test_17_highest_ranking_works_with_normalized_fields(self):
        """17. Both metric and group_by lowercase match real Superstore extraction."""
        claim = Claim(
            subject="West",
            metric="profit",
            value=108418.45,
            comparison="highest",
            group_by="region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)

    def test_18_mismatch_behavior_preserved_with_normalized_fields(self):
        """18. Contradiction ('East' is highest) correctly returns MISMATCH even with lowercase fields."""
        claim = Claim(
            subject="East",
            metric="profit",
            comparison="highest",
            group_by="region",
        )
        result = VerificationService.verify(claim, self.execution_result)
        self.assertEqual(result.status, "mismatch")
        self.assertFalse(result.matched)


if __name__ == "__main__":
    unittest.main()
