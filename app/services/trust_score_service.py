from typing import List

from app.models.trust_score import TrustScore, TrustStatus
from app.models.verification import VerificationResult


class TrustScoreError(Exception):
    """Raised when trust score calculation encounters invalid input."""
    pass


class TrustScoreService:
    """
    Calculates a deterministic, explainable trust score based purely
    on the VerificationResult objects produced by VerificationService.
    Does NOT use an LLM.
    """

    POINTS_VERIFIED = 100.0
    POINTS_INCONCLUSIVE = 50.0
    POINTS_MISMATCH = 0.0

    @classmethod
    def get_status(cls, score: float) -> TrustStatus:
        """
        Determines the trust interpretation category based on score.

        Boundaries:
            90.0–100.0:  "High trust"
            70.0–89.99:  "Moderate trust"
            50.0–69.99:  "Low trust"
            0.0–49.99:   "Very low trust"
        """
        if score >= 90.0:
            return "High trust"
        elif score >= 70.0:
            return "Moderate trust"
        elif score >= 50.0:
            return "Low trust"
        else:
            return "Very low trust"

    @classmethod
    def _build_explanation(
        cls,
        total: int,
        verified: int,
        mismatch: int,
        inconclusive: int
    ) -> str:
        """Generates a deterministic summary explanation of the verification results."""
        if total == 0:
            return "No analytical claims were available for verification."

        base = f"{verified} of {total} analytical claims were verified"
        details = []

        if mismatch > 0:
            details.append(
                f"{mismatch} claim contradicted the dataset evidence"
                if mismatch == 1 else
                f"{mismatch} claims contradicted the dataset evidence"
            )

        if inconclusive > 0:
            details.append(
                f"{inconclusive} claim could not be conclusively evaluated"
                if inconclusive == 1 else
                f"{inconclusive} claims could not be conclusively evaluated"
            )

        if details:
            return f"{base}; " + "; ".join(details) + "."
        return f"{base}."

    @classmethod
    def calculate(cls, verification_results: List[VerificationResult]) -> TrustScore:
        """
        Calculates the deterministic TrustScore from a list of VerificationResult objects.

        Args:
            verification_results: List of VerificationResult models.

        Returns:
            Structured TrustScore model.

        Raises:
            TrustScoreError: When input is not a list or contains non-VerificationResult items.
        """
        if not isinstance(verification_results, list):
            raise TrustScoreError(
                f"Expected a list of VerificationResult, got {type(verification_results).__name__}."
            )

        total_claims = len(verification_results)
        if total_claims == 0:
            return TrustScore(
                score=0.0,
                status="Very low trust",
                total_claims=0,
                verified_claims=0,
                mismatch_claims=0,
                inconclusive_claims=0,
                explanation="No analytical claims were available for verification.",
            )

        verified = 0
        mismatch = 0
        inconclusive = 0
        total_points = 0.0

        for idx, vr in enumerate(verification_results):
            if not isinstance(vr, VerificationResult):
                raise TrustScoreError(
                    f"Item at index {idx} is not a VerificationResult (got {type(vr).__name__})."
                )

            if vr.status == "verified":
                verified += 1
                total_points += cls.POINTS_VERIFIED
            elif vr.status == "inconclusive":
                inconclusive += 1
                total_points += cls.POINTS_INCONCLUSIVE
            elif vr.status == "mismatch":
                mismatch += 1
                total_points += cls.POINTS_MISMATCH
            else:
                raise TrustScoreError(f"Unknown verification status at index {idx}: '{vr.status}'.")

        raw_score = total_points / total_claims
        score = round(raw_score, 2)
        status = cls.get_status(score)
        explanation = cls._build_explanation(total_claims, verified, mismatch, inconclusive)

        return TrustScore(
            score=score,
            status=status,
            total_claims=total_claims,
            verified_claims=verified,
            mismatch_claims=mismatch,
            inconclusive_claims=inconclusive,
            explanation=explanation,
        )
