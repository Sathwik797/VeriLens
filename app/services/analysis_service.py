from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import pandas as pd

from app.eda.profiler import DataProfiler
from app.executor.safe_executor import ExecutionResult, SafeExecutor
from app.models.analysis_plan import AnalysisPlan
from app.models.verification import AnalyticalClaim, VerificationResult
from app.services.claim_extractor_service import ClaimExtractorService
from app.services.narrator_service import NarratorService
from app.services.planner_service import PlannerService
from app.services.verification_service import VerificationService


class AnalysisServiceError(Exception):
    """Raised when analysis orchestration fails due to invalid parameters or empty data."""
    pass


@dataclass
class AnalysisPipelineResult:
    """
    Structured outcome of the end-to-end VeriLens analysis and self-verification pipeline.
    Preserves ground-truth evidence, reasoning plan, narrator explanation,
    extracted claims, and deterministic verification results.
    """
    question: str
    analysis_plan: AnalysisPlan
    execution_result: ExecutionResult
    explanation: str
    claims: List[AnalyticalClaim]
    verification_results: List[VerificationResult]

    @property
    def evidence(self) -> List[Dict[str, Any]]:
        """Convenience accessor for ground-truth execution evidence records."""
        return self.execution_result.to_records()


class AnalysisService:
    """
    Orchestrates the end-to-end self-verifying analytical pipeline:
    1. Dataset Profiling (DataProfiler)
    2. Analytical Planning (PlannerService)
    3. Deterministic Safe Execution (SafeExecutor)
    4. Natural-Language Narration (NarratorService)
    5. Atomic Claim Extraction (ClaimExtractorService)
    6. Deterministic Ground-Truth Verification (VerificationService)
    """

    def __init__(
        self,
        planner_service: Optional[PlannerService] = None,
        narrator_service: Optional[NarratorService] = None,
        claim_extractor_service: Optional[ClaimExtractorService] = None,
    ):
        self.planner_service = planner_service or PlannerService()
        self.narrator_service = narrator_service or NarratorService()
        self.claim_extractor_service = claim_extractor_service or ClaimExtractorService()

    def analyze_question(
        self,
        df: pd.DataFrame,
        question: str,
    ) -> AnalysisPipelineResult:
        """
        Runs the complete self-verification analysis loop for a natural-language question.

        Args:
            df: The pandas DataFrame to query.
            question: The user's natural language question.

        Returns:
            AnalysisPipelineResult with plan, ground truth, narration, claims, and verification.
        """
        if not isinstance(df, pd.DataFrame):
            raise AnalysisServiceError(f"Expected a pandas DataFrame, got {type(df).__name__}.")
        if df.empty:
            raise AnalysisServiceError("Cannot analyze an empty DataFrame.")
        if not question or not question.strip():
            raise AnalysisServiceError("Question cannot be empty.")

        # Step 1: Generate dataset schema profile
        profile = DataProfiler.generate_profile(df)

        # Step 2: Produce validated AnalysisPlan via LLM Planner
        plan = self.planner_service.generate_plan(question=question.strip(), schema=profile)

        # Step 3: Execute plan deterministically via SafeExecutor to obtain ground truth
        execution_result = SafeExecutor.execute(df, plan)

        # Step 4: Generate natural-language explanation from evidence via Narrator
        explanation = self.narrator_service.generate(
            question=question.strip(),
            analysis_plan=plan,
            execution_result=execution_result,
        )

        # Step 5: Extract atomic analytical claims from explanation via ClaimExtractor
        claims = self.claim_extractor_service.extract_claims(
            explanation=explanation,
            question=question.strip(),
        )

        # Step 6: Deterministically verify EACH claim against the ground-truth evidence
        verification_results: List[VerificationResult] = [
            VerificationService.verify(claim, execution_result)
            for claim in claims
        ]

        return AnalysisPipelineResult(
            question=question.strip(),
            analysis_plan=plan,
            execution_result=execution_result,
            explanation=explanation,
            claims=claims,
            verification_results=verification_results,
        )
