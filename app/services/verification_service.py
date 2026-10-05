import math
from typing import Any, Dict, List, Optional, Tuple

from app.executor.safe_executor import ExecutionResult
from app.models.verification import AnalyticalClaim, VerificationResult


class VerificationService:
    """
    Deterministic verification engine that validates AnalyticalClaim objects
    against ground-truth ExecutionResult data.
    """

    # Default tolerances for float comparison (e.g. currency rounding differences)
    DEFAULT_REL_TOL = 0.001   # 0.1% relative tolerance
    DEFAULT_ABS_TOL = 0.05    # 5 cents / 0.05 absolute tolerance

    @classmethod
    def _resolve_column_key(
        cls,
        target_name: Optional[str],
        records: List[Dict[str, Any]]
    ) -> Optional[str]:
        """
        Resolves a claim column/metric name against actual evidence dictionary keys
        in a case-insensitive manner.
        Returns the exact case-sensitive key present in the evidence records, or None if not found.
        Does not mutate the evidence records and does not perform fuzzy matching.
        """
        if not target_name:
            return None

        cleaned_target = str(target_name).strip().lower()

        for r in records:
            for key in r.keys():
                if str(key).strip().lower() == cleaned_target:
                    return key

        return None

    @classmethod
    def verify(
        cls,
        claim: AnalyticalClaim,
        execution_result: ExecutionResult,
        rel_tol: float = DEFAULT_REL_TOL,
        abs_tol: float = DEFAULT_ABS_TOL,
    ) -> VerificationResult:
        """
        Deterministically evaluates an AnalyticalClaim against an ExecutionResult.

        Args:
            claim: The analytical claim to verify.
            execution_result: Ground-truth result from SafeExecutor.
            rel_tol: Relative numerical tolerance for floating-point comparisons.
            abs_tol: Absolute numerical tolerance for floating-point comparisons.

        Returns:
            VerificationResult with status ('verified', 'mismatch', 'inconclusive'),
            boolean matched flag, actual vs expected values, evidence records, and reason.
        """
        records = execution_result.to_records()

        # 1. Guard against empty evidence
        if not records:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=None,
                evidence=[],
                reason="ExecutionResult contains no evidence records.",
            )

        # 2. Resolve metric against evidence keys (case-insensitively)
        resolved_metric = cls._resolve_column_key(claim.metric, records)
        if resolved_metric is None:
            # Check scalar result fallback
            if execution_result.scalar_value is None:
                return VerificationResult(
                    status="inconclusive",
                    matched=False,
                    expected_value=claim.value,
                    actual_value=None,
                    evidence=records,
                    reason=f"Metric '{claim.metric}' not found in evidence records.",
                )
            resolved_metric = claim.metric

        # 3. Resolve group_by against evidence keys (case-insensitively if specified)
        resolved_group_by = (
            cls._resolve_column_key(claim.group_by, records)
            if claim.group_by else None
        )

        # 4. Route by comparison type
        comparison = claim.comparison or "equal"

        if comparison in ("highest", "lowest"):
            return cls._verify_ranking(
                claim=claim,
                records=records,
                ranking_type=comparison,
                resolved_metric=resolved_metric,
                resolved_group_by=resolved_group_by,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        elif comparison == "equal":
            return cls._verify_equality(
                claim=claim,
                execution_result=execution_result,
                records=records,
                resolved_metric=resolved_metric,
                resolved_group_by=resolved_group_by,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        elif comparison in ("greater_than", "less_than", "greater_than_or_equal", "less_than_or_equal"):
            return cls._verify_inequality(
                claim=claim,
                execution_result=execution_result,
                records=records,
                comparison=comparison,
                resolved_metric=resolved_metric,
                resolved_group_by=resolved_group_by,
                rel_tol=rel_tol,
                abs_tol=abs_tol,
            )
        else:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=None,
                evidence=records,
                reason=f"Unsupported comparison type: '{comparison}'.",
            )

    @classmethod
    def _is_numeric_close(
        cls,
        val1: Any,
        val2: Any,
        rel_tol: float,
        abs_tol: float
    ) -> bool:
        """Compares two numeric values within configured tolerances."""
        try:
            f1, f2 = float(val1), float(val2)
            return math.isclose(f1, f2, rel_tol=rel_tol, abs_tol=abs_tol)
        except (ValueError, TypeError):
            return val1 == val2

    @classmethod
    def _find_subject_row(
        cls,
        subject: str,
        group_by: Optional[str],
        records: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """Locates the evidence record corresponding to the claimed subject."""
        target = str(subject).strip().lower()

        for r in records:
            if group_by and group_by in r:
                if str(r[group_by]).strip().lower() == target:
                    return r
            else:
                for k, v in r.items():
                    if str(v).strip().lower() == target:
                        return r
        return None

    @classmethod
    def _verify_ranking(
        cls,
        claim: AnalyticalClaim,
        records: List[Dict[str, Any]],
        ranking_type: str,
        resolved_metric: str,
        resolved_group_by: Optional[str],
        rel_tol: float,
        abs_tol: float,
    ) -> VerificationResult:
        """Verifies 'highest' or 'lowest' claims using resolved column names."""
        if not claim.subject:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=None,
                evidence=records,
                reason=f"A subject is required to verify a '{ranking_type}' claim.",
            )

        # Extract numeric values for the metric using the resolved evidence key
        numeric_rows = [r for r in records if isinstance(r.get(resolved_metric), (int, float))]
        if not numeric_rows:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=None,
                evidence=records,
                reason=f"No numeric values available for metric '{claim.metric}'.",
            )

        subject_row = cls._find_subject_row(claim.subject, resolved_group_by, numeric_rows)
        if subject_row is None:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=None,
                evidence=records,
                reason=f"Subject '{claim.subject}' was not found in the evidence records.",
            )

        subject_metric_val = subject_row[resolved_metric]

        if ranking_type == "highest":
            extreme_val = max(r[resolved_metric] for r in numeric_rows)
            extreme_rows = [
                r for r in numeric_rows
                if cls._is_numeric_close(r[resolved_metric], extreme_val, rel_tol, abs_tol)
            ]
            is_extreme = any(
                cls._find_subject_row(claim.subject, resolved_group_by, [r]) is not None
                for r in extreme_rows
            )
            rank_label = "highest"
        else:
            extreme_val = min(r[resolved_metric] for r in numeric_rows)
            extreme_rows = [
                r for r in numeric_rows
                if cls._is_numeric_close(r[resolved_metric], extreme_val, rel_tol, abs_tol)
            ]
            is_extreme = any(
                cls._find_subject_row(claim.subject, resolved_group_by, [r]) is not None
                for r in extreme_rows
            )
            rank_label = "lowest"

        # Check ranking contradiction
        if not is_extreme:
            return VerificationResult(
                status="mismatch",
                matched=False,
                expected_value=claim.value,
                actual_value=subject_metric_val,
                evidence=records,
                reason=(
                    f"Subject '{claim.subject}' does not have the {rank_label} '{claim.metric}'. "
                    f"Actual {rank_label} is {extreme_val}, whereas '{claim.subject}' has {subject_metric_val}."
                ),
            )

        # Check numeric value match (if specified in claim)
        if claim.value is not None:
            value_matches = cls._is_numeric_close(claim.value, subject_metric_val, rel_tol, abs_tol)
            if not value_matches:
                return VerificationResult(
                    status="mismatch",
                    matched=False,
                    expected_value=claim.value,
                    actual_value=subject_metric_val,
                    evidence=records,
                    reason=(
                        f"Subject '{claim.subject}' is the {rank_label} in '{claim.metric}', "
                        f"but claimed value ({claim.value}) does not match actual value ({subject_metric_val})."
                    ),
                )

        return VerificationResult(
            status="verified",
            matched=True,
            expected_value=claim.value,
            actual_value=subject_metric_val,
            evidence=records,
            reason=f"Claim verified: '{claim.subject}' has the {rank_label} '{claim.metric}' ({subject_metric_val}).",
        )

    @classmethod
    def _extract_actual_value(
        cls,
        claim: AnalyticalClaim,
        execution_result: ExecutionResult,
        records: List[Dict[str, Any]],
        resolved_metric: str,
        resolved_group_by: Optional[str],
    ) -> Tuple[Optional[Any], Optional[str]]:
        """Resolves the actual value from subject row or scalar result."""
        if claim.subject:
            subject_row = cls._find_subject_row(claim.subject, resolved_group_by, records)
            if subject_row is None:
                return None, f"Subject '{claim.subject}' not found in evidence."
            if resolved_metric not in subject_row:
                return None, f"Metric '{claim.metric}' not found in subject record."
            return subject_row[resolved_metric], None

        scalar = execution_result.scalar_value
        if scalar is not None:
            return scalar, None

        if len(records) == 1 and resolved_metric in records[0]:
            return records[0][resolved_metric], None

        return None, "Claim has no subject and evidence does not represent a single scalar value."

    @classmethod
    def _verify_equality(
        cls,
        claim: AnalyticalClaim,
        execution_result: ExecutionResult,
        records: List[Dict[str, Any]],
        resolved_metric: str,
        resolved_group_by: Optional[str],
        rel_tol: float,
        abs_tol: float,
    ) -> VerificationResult:
        """Verifies 'equal' comparison claims using resolved column names."""
        if claim.value is None:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=None,
                actual_value=None,
                evidence=records,
                reason="Equality claim requires a claimed 'value' to compare against.",
            )

        actual_val, err_msg = cls._extract_actual_value(
            claim=claim,
            execution_result=execution_result,
            records=records,
            resolved_metric=resolved_metric,
            resolved_group_by=resolved_group_by,
        )
        if err_msg:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=None,
                evidence=records,
                reason=err_msg,
            )

        matched = cls._is_numeric_close(claim.value, actual_val, rel_tol, abs_tol)
        status = "verified" if matched else "mismatch"
        reason = (
            f"Value matches evidence ({actual_val})."
            if matched else
            f"Claimed value ({claim.value}) does not match actual value ({actual_val})."
        )

        return VerificationResult(
            status=status,
            matched=matched,
            expected_value=claim.value,
            actual_value=actual_val,
            evidence=records,
            reason=reason,
        )

    @classmethod
    def _verify_inequality(
        cls,
        claim: AnalyticalClaim,
        execution_result: ExecutionResult,
        records: List[Dict[str, Any]],
        comparison: str,
        resolved_metric: str,
        resolved_group_by: Optional[str],
        rel_tol: float,
        abs_tol: float,
    ) -> VerificationResult:
        """Verifies inequality claims (> , < , >= , <=) using resolved column names."""
        if claim.value is None:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=None,
                actual_value=None,
                evidence=records,
                reason=f"Inequality '{comparison}' requires a threshold 'value' to compare against.",
            )

        actual_val, err_msg = cls._extract_actual_value(
            claim=claim,
            execution_result=execution_result,
            records=records,
            resolved_metric=resolved_metric,
            resolved_group_by=resolved_group_by,
        )
        if err_msg:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=None,
                evidence=records,
                reason=err_msg,
            )

        try:
            f_actual = float(actual_val)
            f_claim = float(claim.value)
        except (ValueError, TypeError):
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=actual_val,
                evidence=records,
                reason="Inequality comparisons require numeric values.",
            )

        is_close = math.isclose(f_actual, f_claim, rel_tol=rel_tol, abs_tol=abs_tol)

        if comparison == "greater_than":
            matched = f_actual > f_claim and not is_close
            op_symbol = ">"
        elif comparison == "less_than":
            matched = f_actual < f_claim and not is_close
            op_symbol = "<"
        elif comparison == "greater_than_or_equal":
            matched = f_actual >= f_claim or is_close
            op_symbol = ">="
        elif comparison == "less_than_or_equal":
            matched = f_actual <= f_claim or is_close
            op_symbol = "<="
        else:
            return VerificationResult(
                status="inconclusive",
                matched=False,
                expected_value=claim.value,
                actual_value=actual_val,
                evidence=records,
                reason=f"Unknown comparison: '{comparison}'.",
            )

        status = "verified" if matched else "mismatch"
        reason = (
            f"Claim verified: actual value ({actual_val}) {op_symbol} claimed threshold ({claim.value})."
            if matched else
            f"Claim contradiction: actual value ({actual_val}) is not {op_symbol} claimed threshold ({claim.value})."
        )

        return VerificationResult(
            status=status,
            matched=matched,
            expected_value=claim.value,
            actual_value=actual_val,
            evidence=records,
            reason=reason,
        )
