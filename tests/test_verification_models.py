import unittest
from pydantic import ValidationError

from app.models.verification import AnalyticalClaim, Claim, VerificationResult


class TestVerificationModels(unittest.TestCase):

    def test_01_valid_claim_creation(self):
        """Test: Valid generic analytical claim."""
        claim = AnalyticalClaim(
            subject="Technology",
            metric="Sales",
            value=836154.03,
            comparison="equal",
            group_by="Category",
        )
        self.assertEqual(claim.subject, "Technology")
        self.assertEqual(claim.metric, "Sales")
        self.assertEqual(claim.value, 836154.03)
        self.assertEqual(claim.comparison, "equal")
        self.assertEqual(claim.group_by, "Category")

    def test_02_valid_highest_profit_claim(self):
        """Test: Valid highest-profit claim matching project milestone example."""
        claim = Claim(
            subject="West",
            metric="Profit",
            value=108418.45,
            comparison="highest",
            group_by="Region",
        )
        data = claim.model_dump()
        self.assertEqual(data["subject"], "West")
        self.assertEqual(data["metric"], "Profit")
        self.assertEqual(data["value"], 108418.45)
        self.assertEqual(data["comparison"], "highest")
        self.assertEqual(data["group_by"], "Region")

    def test_03_invalid_comparison_rejected(self):
        """Test: Uncontrolled or arbitrary comparison strings must be rejected."""
        with self.assertRaises(ValidationError) as ctx:
            AnalyticalClaim(
                metric="Profit",
                comparison="most_profitable",  # Not in allowed ComparisonType
            )
        self.assertIn("comparison", str(ctx.exception))

    def test_04_valid_verified_result(self):
        """Test: Valid VerificationResult with 'verified' status and structured evidence."""
        result = VerificationResult(
            status="verified",
            matched=True,
            expected_value=108418.45,
            actual_value=108418.4489,
            evidence=[{"Region": "West", "Profit": 108418.4489}],
            reason="The claimed value and ranking match the deterministic dataset result.",
        )
        self.assertEqual(result.status, "verified")
        self.assertTrue(result.matched)
        self.assertEqual(len(result.evidence), 1)
        self.assertEqual(result.evidence[0]["Region"], "West")

    def test_05_valid_mismatch_result(self):
        """Test: Valid VerificationResult with 'mismatch' status."""
        result = VerificationResult(
            status="mismatch",
            matched=False,
            expected_value=91523.00,
            actual_value=108418.4489,
            evidence=[{"Region": "West", "Profit": 108418.4489}, {"Region": "East", "Profit": 91523.00}],
            reason="East is not the highest profit region; West has higher profit.",
        )
        self.assertEqual(result.status, "mismatch")
        self.assertFalse(result.matched)
        self.assertIn("East is not the highest", result.reason)

    def test_06_valid_inconclusive_result(self):
        """Test: Valid VerificationResult with 'inconclusive' status."""
        result = VerificationResult(
            status="inconclusive",
            matched=False,
            expected_value=None,
            actual_value=None,
            evidence=None,
            reason="Dataset does not contain necessary temporal granularity to verify quarterly claim.",
        )
        self.assertEqual(result.status, "inconclusive")
        self.assertFalse(result.matched)
        self.assertIsNone(result.actual_value)

    def test_07_invalid_verification_status_rejected(self):
        """Test: Uncontrolled status strings must be rejected."""
        with self.assertRaises(ValidationError) as ctx:
            VerificationResult(
                status="partially_true",  # Not in allowed VerificationStatus
                matched=True,
            )
        self.assertIn("status", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
