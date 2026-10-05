import unittest

from app.models.trust_score import TrustScore
from app.models.verification import VerificationResult
from app.services.trust_score_service import TrustScoreError, TrustScoreService


class TestTrustScoreService(unittest.TestCase):

    def _make_vr(self, status: str, matched: bool = True) -> VerificationResult:
        """Helper to create minimal valid VerificationResult objects."""
        return VerificationResult(
            status=status,
            matched=matched,
            expected_value=100.0,
            actual_value=100.0 if matched else 50.0,
            evidence=[{"col": 100.0}],
            reason=f"Status is {status}",
        )

    def test_01_one_verified_claim_score_100(self):
        """1. One verified claim -> score 100.0, 'High trust'."""
        vr = self._make_vr("verified", matched=True)
        res = TrustScoreService.calculate([vr])

        self.assertIsInstance(res, TrustScore)
        self.assertEqual(res.score, 100.0)
        self.assertEqual(res.status, "High trust")
        self.assertEqual(res.total_claims, 1)
        self.assertEqual(res.verified_claims, 1)
        self.assertEqual(res.mismatch_claims, 0)
        self.assertEqual(res.inconclusive_claims, 0)
        self.assertEqual(res.explanation, "1 of 1 analytical claims were verified.")

    def test_02_one_mismatch_claim_score_0(self):
        """2. One mismatch claim -> score 0.0, 'Very low trust'."""
        vr = self._make_vr("mismatch", matched=False)
        res = TrustScoreService.calculate([vr])

        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.status, "Very low trust")
        self.assertEqual(res.total_claims, 1)
        self.assertEqual(res.mismatch_claims, 1)
        self.assertEqual(
            res.explanation,
            "0 of 1 analytical claims were verified; 1 claim contradicted the dataset evidence."
        )

    def test_03_one_inconclusive_claim_score_50(self):
        """3. One inconclusive claim -> score 50.0, 'Low trust'."""
        vr = self._make_vr("inconclusive", matched=False)
        res = TrustScoreService.calculate([vr])

        self.assertEqual(res.score, 50.0)
        self.assertEqual(res.status, "Low trust")
        self.assertEqual(res.total_claims, 1)
        self.assertEqual(res.inconclusive_claims, 1)
        self.assertEqual(
            res.explanation,
            "0 of 1 analytical claims were verified; 1 claim could not be conclusively evaluated."
        )

    def test_04_verified_and_mismatch_score_50(self):
        """4. 1 verified + 1 mismatch -> score (100 + 0) / 2 = 50.0."""
        vr1 = self._make_vr("verified", matched=True)
        vr2 = self._make_vr("mismatch", matched=False)
        res = TrustScoreService.calculate([vr1, vr2])

        self.assertEqual(res.score, 50.0)
        self.assertEqual(res.status, "Low trust")
        self.assertEqual(res.total_claims, 2)
        self.assertEqual(res.verified_claims, 1)
        self.assertEqual(res.mismatch_claims, 1)
        self.assertEqual(
            res.explanation,
            "1 of 2 analytical claims were verified; 1 claim contradicted the dataset evidence."
        )

    def test_05_verified_and_inconclusive_score_75(self):
        """5. 1 verified + 1 inconclusive -> score (100 + 50) / 2 = 75.0, 'Moderate trust'."""
        vr1 = self._make_vr("verified", matched=True)
        vr2 = self._make_vr("inconclusive", matched=False)
        res = TrustScoreService.calculate([vr1, vr2])

        self.assertEqual(res.score, 75.0)
        self.assertEqual(res.status, "Moderate trust")
        self.assertEqual(res.total_claims, 2)
        self.assertEqual(res.verified_claims, 1)
        self.assertEqual(res.inconclusive_claims, 1)
        self.assertEqual(
            res.explanation,
            "1 of 2 analytical claims were verified; 1 claim could not be conclusively evaluated."
        )

    def test_06_verified_and_verified_score_100(self):
        """6. 2 verified claims -> score (100 + 100) / 2 = 100.0."""
        vr1 = self._make_vr("verified", matched=True)
        vr2 = self._make_vr("verified", matched=True)
        res = TrustScoreService.calculate([vr1, vr2])

        self.assertEqual(res.score, 100.0)
        self.assertEqual(res.status, "High trust")
        self.assertEqual(res.total_claims, 2)
        self.assertEqual(res.verified_claims, 2)
        self.assertEqual(res.explanation, "2 of 2 analytical claims were verified.")

    def test_07_no_claims_score_0_with_clear_explanation(self):
        """7. Empty claims list -> score 0.0 with clear non-crashing explanation."""
        res = TrustScoreService.calculate([])

        self.assertEqual(res.score, 0.0)
        self.assertEqual(res.status, "Very low trust")
        self.assertEqual(res.total_claims, 0)
        self.assertEqual(res.explanation, "No analytical claims were available for verification.")

    def test_08_multiple_mixed_claims(self):
        """8. Mixed: 2 verified + 1 inconclusive + 1 mismatch -> (200 + 50 + 0) / 4 = 62.5."""
        vr_list = [
            self._make_vr("verified", True),
            self._make_vr("verified", True),
            self._make_vr("inconclusive", False),
            self._make_vr("mismatch", False),
        ]
        res = TrustScoreService.calculate(vr_list)

        self.assertEqual(res.score, 62.5)
        self.assertEqual(res.status, "Low trust")
        self.assertEqual(res.total_claims, 4)
        self.assertEqual(res.verified_claims, 2)
        self.assertEqual(res.mismatch_claims, 1)
        self.assertEqual(res.inconclusive_claims, 1)
        self.assertIn("2 of 4 analytical claims were verified", res.explanation)
        self.assertIn("1 claim contradicted the dataset evidence", res.explanation)
        self.assertIn("1 claim could not be conclusively evaluated", res.explanation)

    def test_09_status_interpretation_boundaries(self):
        """9. Verifies exact category boundaries: High, Moderate, Low, Very low trust."""
        self.assertEqual(TrustScoreService.get_status(100.0), "High trust")
        self.assertEqual(TrustScoreService.get_status(90.0), "High trust")
        self.assertEqual(TrustScoreService.get_status(89.99), "Moderate trust")
        self.assertEqual(TrustScoreService.get_status(70.0), "Moderate trust")
        self.assertEqual(TrustScoreService.get_status(69.99), "Low trust")
        self.assertEqual(TrustScoreService.get_status(50.0), "Low trust")
        self.assertEqual(TrustScoreService.get_status(49.99), "Very low trust")
        self.assertEqual(TrustScoreService.get_status(0.0), "Very low trust")

    def test_10_invalid_inputs_raise_clear_error(self):
        """10. Non-list and list containing non-VerificationResult items raise TrustScoreError."""
        with self.assertRaises(TrustScoreError):
            TrustScoreService.calculate("not a list")  # type: ignore

        with self.assertRaises(TrustScoreError):
            TrustScoreService.calculate([self._make_vr("verified"), "invalid_item"])  # type: ignore


if __name__ == "__main__":
    unittest.main()
