from dataclasses import dataclass
from datetime import datetime, timezone
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.eda.column_normalizer import ColumnNormalizer
from app.models.join_execution_result import JoinExecutionResult, JoinExecutionStatus
from app.models.join_recommendation import (
    JoinCardinality,
    JoinRecommendation,
    RecommendedJoinType,
)

logger = logging.getLogger(__name__)

# Strict mapping of controlled join types to pandas how parameter
_JOIN_TYPE_TO_PANDAS: Dict[str, str] = {
    "LEFT_JOIN": "left",
    "INNER_JOIN": "inner",
    "RIGHT_JOIN": "right",
    "FULL_OUTER_JOIN": "outer",
}


@dataclass
class SafeJoinExecutionOutput:
    """
    Container for the outcome of SafeJoinExecutor.
    Keeps the structured metadata separate from the raw pandas DataFrame.
    """
    result: JoinExecutionResult
    dataframe: Optional[pd.DataFrame] = None


class SafeJoinExecutor:
    """
    Deterministic Safe Join Executor for VeriLens AI.
    Executes pandas joins ONLY after Phase 2 Join Intelligence approval.
    Enforces strict safety gates, preserves input DataFrame immutability,
    validates post-join results, and tracks complete data provenance.
    """

    @classmethod
    def _extract_dataframe_provenance(cls, df: pd.DataFrame) -> Optional[str]:
        """Extracts authoritative dataset identity from DataFrame attributes."""
        if hasattr(df, "attrs") and isinstance(df.attrs, dict):
            for key in ("filename", "name", "dataset_name", "orig_name", "source_name", "file_name"):
                val = df.attrs.get(key)
                if val and isinstance(val, str) and val.strip():
                    return os.path.basename(val.strip())
        return None

    @classmethod
    def _resolve_authoritative_identity(
        cls, df: pd.DataFrame, explicit_name: Optional[str] = None
    ) -> Tuple[Optional[str], Optional[str]]:
        """
        Determines authoritative dataset identity with anti-spoofing enforcement.
        1. DataFrame provenance (df.attrs["filename"]) is primary and authoritative.
        2. If explicit_name contradicts DataFrame provenance, an error is returned.
        3. If DataFrame has NO provenance and no explicit name, returns (None, None).
        4. If DataFrame has NO provenance but explicit_name is provided, explicit_name is used.
        """
        df_provenance = cls._extract_dataframe_provenance(df)
        explicit_norm = (
            os.path.basename(str(explicit_name).strip())
            if explicit_name and str(explicit_name).strip()
            else None
        )

        if df_provenance and explicit_norm:
            if df_provenance.lower() != explicit_norm.lower():
                return None, (
                    f"Contradictory dataset provenance detected: DataFrame provenance is '{df_provenance}' "
                    f"but caller specified '{explicit_norm}'. Potential identity spoofing blocked."
                )
            return df_provenance, None

        if df_provenance:
            return df_provenance, None

        if explicit_norm:
            return explicit_norm, None

        return None, None

    @classmethod
    def execute(
        cls,
        left_df: pd.DataFrame,
        right_df: pd.DataFrame,
        recommendation: JoinRecommendation,
        requested_join_type: Optional[str] = None,
        user_approved: bool = True,
        left_name: Optional[str] = None,
        right_name: Optional[str] = None,
    ) -> SafeJoinExecutionOutput:
        """
        Executes a safe join based on the approved Phase 2 recommendation.
        Returns SafeJoinExecutionOutput containing the Pydantic result model
        and the derived joined DataFrame (or None if blocked).
        """
        name_a = recommendation.dataset_a or "left_dataset"
        name_b = recommendation.dataset_b or "right_dataset"
        col_left = recommendation.left_column
        col_right = recommendation.right_column
        now_ts = datetime.now(timezone.utc).isoformat()

        # Derive clean dataset name for the resulting derived table
        stem_a = os.path.splitext(os.path.basename(name_a))[0]
        stem_b = os.path.splitext(os.path.basename(name_b))[0]
        derived_name = f"{stem_a}_{stem_b}_joined"

        # -------------------------------------------------------------
        # 1. SAFETY GATES (CRITICAL PRE-JOIN ENFORCEMENT)
        # -------------------------------------------------------------
        # Gate 1: Recommendation exists and is valid
        if recommendation is None:
            return cls._block_execution(
                error_reason="Missing Phase 2 JoinRecommendation.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Gate 2: Recommendation cannot be NO_SAFE_JOIN
        if recommendation.recommended_join == "NO_SAFE_JOIN":
            return cls._block_execution(
                error_reason=f"Join recommendation is NO_SAFE_JOIN: {recommendation.explanation or 'No safe key found.'}",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Gate 3: Safe to execute flag must be True
        if not recommendation.safe_to_execute:
            return cls._block_execution(
                error_reason=f"Phase 2 flagged join as unsafe (safe_to_execute=False): {recommendation.explanation or 'Safety rules violated.'}",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Gate 4: User approval check
        if not user_approved:
            return cls._block_execution(
                error_reason="Join execution requires explicit user approval.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Gate 5: Deterministic Dataset Identity & Provenance Validation (Anti-Spoofing)
        id_left, err_left = cls._resolve_authoritative_identity(left_df, left_name)
        id_right, err_right = cls._resolve_authoritative_identity(right_df, right_name)

        if err_left:
            return cls._block_execution(
                error_reason=f"Dataset identity mismatch: {err_left}",
                recommendation=recommendation,
                name_a=left_name or name_a,
                name_b=right_name or name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        if err_right:
            return cls._block_execution(
                error_reason=f"Dataset identity mismatch: {err_right}",
                recommendation=recommendation,
                name_a=left_name or name_a,
                name_b=right_name or name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        target_a = os.path.basename(str(recommendation.dataset_a).strip()) if recommendation and recommendation.dataset_a else ""
        target_b = os.path.basename(str(recommendation.dataset_b).strip()) if recommendation and recommendation.dataset_b else ""

        if not id_left or not id_right:
            missing_side = "left and right" if (not id_left and not id_right) else ("left" if not id_left else "right")
            return cls._block_execution(
                error_reason=f"Dataset identity validation failed: reliable provenance could not be established for {missing_side} DataFrame. Join execution blocked for safety.",
                recommendation=recommendation,
                name_a=id_left or name_a,
                name_b=id_right or name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        matches_canonical = (id_left.lower() == target_a.lower()) and (id_right.lower() == target_b.lower())
        matches_reversed = (id_left.lower() == target_b.lower()) and (id_right.lower() == target_a.lower())

        if not matches_canonical and not matches_reversed:
            if id_left.lower() != target_a.lower() and id_left.lower() != target_b.lower():
                err_detail = f"Left DataFrame '{id_left}' does not match recommendation dataset_a '{target_a}'."
            elif id_right.lower() != target_b.lower() and id_right.lower() != target_a.lower():
                err_detail = f"Right DataFrame '{id_right}' does not match recommendation dataset_b '{target_b}'."
            else:
                err_detail = f"DataFrames ('{id_left}', '{id_right}') do not match recommendation target datasets ('{target_a}', '{target_b}')."

            return cls._block_execution(
                error_reason=f"Dataset identity mismatch: {err_detail} Join execution blocked.",
                recommendation=recommendation,
                name_a=id_left,
                name_b=id_right,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Update confirmed names
        name_a = id_left
        name_b = id_right

        # Gate 6: Requested join type check
        active_join_type = requested_join_type or recommendation.recommended_join
        if active_join_type not in _JOIN_TYPE_TO_PANDAS:
            return cls._block_execution(
                error_reason=f"Unsupported join type: '{active_join_type}'. Allowed types: LEFT_JOIN, INNER_JOIN, RIGHT_JOIN, FULL_OUTER_JOIN.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # If user explicitly requested a join type different from Phase 2 recommendation,
        # ensure it is safe (e.g., INNER_JOIN when LEFT_JOIN was recommended is safe; arbitrary is not)
        if requested_join_type and requested_join_type != recommendation.recommended_join:
            # Allow narrowing to INNER_JOIN or compatible join types
            if requested_join_type not in ("INNER_JOIN", "LEFT_JOIN", "RIGHT_JOIN", "FULL_OUTER_JOIN"):
                return cls._block_execution(
                    error_reason=f"Requested join type '{requested_join_type}' does not match approved '{recommendation.recommended_join}'.",
                    recommendation=recommendation,
                    name_a=name_a,
                    name_b=name_b,
                    derived_name=derived_name,
                    now_ts=now_ts,
                )

        # Gate 6 & 7: Column existence and match with validated candidate
        if col_left not in left_df.columns:
            return cls._block_execution(
                error_reason=f"Left join column '{col_left}' does not exist in left dataset '{name_a}'.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        if col_right not in right_df.columns:
            return cls._block_execution(
                error_reason=f"Right join column '{col_right}' does not exist in right dataset '{name_b}'.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Gate 8: Disallow Many-to-Many joins to prevent Cartesian explosions
        if recommendation.cardinality == "MANY_TO_MANY":
            return cls._block_execution(
                error_reason="Many-to-many join is blocked due to Cartesian explosion risk.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Verify directly that both sides are not duplicated on the key
        u_left = left_df[col_left].nunique()
        u_right = right_df[col_right].nunique()
        n_left = len(left_df)
        n_right = len(right_df)
        if (u_left < n_left) and (u_right < n_right):
            return cls._block_execution(
                error_reason="Direct data check confirmed both datasets contain duplicate keys (many-to-many). Merge blocked.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Gate 9: Key columns must not be index-like
        if ColumnNormalizer.is_index_like(col_left, left_df[col_left]):
            return cls._block_execution(
                error_reason=f"Left key column '{col_left}' is an index-like or row counter column. Join blocked.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        if ColumnNormalizer.is_index_like(col_right, right_df[col_right]):
            return cls._block_execution(
                error_reason=f"Right key column '{col_right}' is an index-like or row counter column. Join blocked.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # Gate 10: Phase 2 confidence check (must be at least 50 / LOW)
        if recommendation.confidence < 50 or recommendation.confidence_level == "VERY LOW":
            return cls._block_execution(
                error_reason=f"Phase 2 confidence is too low ({recommendation.confidence}%, {recommendation.confidence_level}). Join blocked.",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # -------------------------------------------------------------
        # 2. PRE-JOIN AUDIT & IMMUTABILITY CHECK
        # -------------------------------------------------------------
        cols_before_left = len(left_df.columns)
        cols_before_right = len(right_df.columns)
        cols_before_total = cols_before_left + cols_before_right

        pandas_how = _JOIN_TYPE_TO_PANDAS[active_join_type]

        # Check overlapping non-key columns that will receive suffixes
        overlap_non_key = (set(left_df.columns).intersection(set(right_df.columns))) - {col_left, col_right}

        pre_evidence = [
            f"Pre-join audit: '{name_a}' has {n_left:,} rows, {cols_before_left} cols; '{name_b}' has {n_right:,} rows, {cols_before_right} cols.",
            f"Validated key linkage: '{name_a}.{col_left}' <-> '{name_b}.{col_right}'.",
            f"Executing {active_join_type} (pandas how='{pandas_how}').",
        ]
        if overlap_non_key:
            pre_evidence.append(
                f"Disambiguating {len(overlap_non_key)} overlapping non-key column(s) with suffixes '_left' and '_right': {', '.join(sorted(overlap_non_key))}."
            )

        # -------------------------------------------------------------
        # 3. EXECUTE PANDAS MERGE (FIRST & ONLY PLACE MERGE IS CALLED)
        # -------------------------------------------------------------
        # We do not mutate left_df or right_df in place.
        # pandas.merge creates and returns a brand-new DataFrame.
        try:
            joined_df = pd.merge(
                left_df,
                right_df,
                left_on=col_left,
                right_on=col_right,
                how=pandas_how,
                suffixes=("_left", "_right"),
            )
        except Exception as e:
            return cls._block_execution(
                error_reason=f"Pandas merge execution failed: {e}",
                recommendation=recommendation,
                name_a=name_a,
                name_b=name_b,
                derived_name=derived_name,
                now_ts=now_ts,
            )

        # -------------------------------------------------------------
        # 4. POST-JOIN VALIDATION & ANOMALY DETECTION
        # -------------------------------------------------------------
        rows_after = len(joined_df)
        cols_after = len(joined_df.columns)
        max_parent_rows = max(n_left, n_right, 1)
        row_mult_factor = round(rows_after / max_parent_rows, 4)

        warnings: List[str] = []
        duplicate_expansion_detected = False

        # Post-join Check 1: Zero rows outcome
        if rows_after == 0:
            warnings.append("Joined dataset contains zero rows. No records satisfied the join predicate.")

        # Post-join Check 2: Row explosion / duplicate expansion
        if active_join_type in ("LEFT_JOIN", "INNER_JOIN", "RIGHT_JOIN", "FULL_OUTER_JOIN"):
            if recommendation.cardinality == "ONE_TO_MANY" and rows_after > n_left:
                duplicate_expansion_detected = True
            elif recommendation.cardinality == "MANY_TO_ONE" and rows_after > n_right:
                duplicate_expansion_detected = True
            elif recommendation.cardinality == "ONE_TO_ONE" and rows_after > max(n_left, n_right):
                duplicate_expansion_detected = True
                warnings.append(
                    f"Unexpected row expansion in 1:1 join: result has {rows_after:,} rows vs max input {max_parent_rows:,} (factor {row_mult_factor}x)."
                )

            if row_mult_factor > 2.0:
                warnings.append(
                    f"Significant row expansion detected: result grew to {rows_after:,} rows (factor {row_mult_factor}x)."
                )

        # Post-join Check 3: Value match coverage verification
        s_left = left_df[col_left].dropna().astype(str).str.strip()
        s_right = right_df[col_right].dropna().astype(str).str.strip()
        distinct_left = set(s_left.unique())
        distinct_right = set(s_right.unique())

        matched_in_left = int(s_left.isin(distinct_right).sum()) if distinct_right else 0
        unmatched_left = n_left - matched_in_left

        matched_in_right = int(s_right.isin(distinct_left).sum()) if distinct_left else 0
        unmatched_right = n_right - matched_in_right

        if active_join_type == "LEFT_JOIN":
            matched_rows = matched_in_left
        elif active_join_type == "RIGHT_JOIN":
            matched_rows = matched_in_right
        else:
            matched_rows = min(matched_in_left, matched_in_right)

        if matched_rows == 0 and rows_after > 0:
            warnings.append("Zero matched records between key columns. All non-driving attributes filled with nulls.")


        if unmatched_left > 0 and active_join_type in ("INNER_JOIN", "RIGHT_JOIN"):
            warnings.append(f"{unmatched_left:,} rows from left dataset '{name_a}' were excluded by {active_join_type}.")
        if unmatched_right > 0 and active_join_type in ("INNER_JOIN", "LEFT_JOIN"):
            warnings.append(f"{unmatched_right:,} rows from right dataset '{name_b}' were excluded by {active_join_type}.")

        # Determine status
        status: JoinExecutionStatus = "SUCCESS_WITH_WARNINGS" if warnings else "SUCCESS"

        post_evidence = list(pre_evidence)
        post_evidence.append(
            f"Post-join validation: produced {rows_after:,} rows and {cols_after} columns (multiplication factor: {row_mult_factor}x)."
        )
        post_evidence.append(
            f"Match summary: {matched_rows:,} matched rows, {unmatched_left:,} unmatched in '{name_a}', {unmatched_right:,} unmatched in '{name_b}'."
        )

        # Attach provenance to the new DataFrame
        joined_df.attrs["filename"] = derived_name
        joined_df.attrs["provenance"] = {
            "source_dataset_a": name_a,
            "source_dataset_b": name_b,
            "left_key": col_left,
            "right_key": col_right,
            "join_type": active_join_type,
            "timestamp": now_ts,
            "cardinality": recommendation.cardinality,
            "confidence": recommendation.confidence,
        }

        result = JoinExecutionResult(
            status=status,
            safe_to_execute=True,
            left_dataset=name_a,
            right_dataset=name_b,
            left_column=col_left,
            right_column=col_right,
            join_type=active_join_type,
            cardinality=recommendation.cardinality,
            confidence=recommendation.confidence,
            rows_before_left=n_left,
            rows_before_right=n_right,
            rows_after=rows_after,
            columns_before=cols_before_total,
            columns_after=cols_after,
            matched_rows=matched_rows,
            unmatched_left_rows=unmatched_left,
            unmatched_right_rows=unmatched_right,
            row_multiplication_factor=row_mult_factor,
            duplicate_expansion_detected=duplicate_expansion_detected,
            derived_dataset_name=derived_name,
            warnings=warnings,
            evidence=post_evidence,
            error_reason=None,
            execution_timestamp=now_ts,
        )

        return SafeJoinExecutionOutput(result=result, dataframe=joined_df)

    @classmethod
    def _block_execution(
        cls,
        error_reason: str,
        recommendation: Optional[JoinRecommendation],
        name_a: str,
        name_b: str,
        derived_name: str,
        now_ts: str,
    ) -> SafeJoinExecutionOutput:
        """
        Creates a structured BLOCKED execution result without calling pandas.merge.
        """
        logger.warning(f"SafeJoinExecutor BLOCKED join execution: {error_reason}")

        col_left = recommendation.left_column if recommendation else ""
        col_right = recommendation.right_column if recommendation else ""
        cardinality: JoinCardinality = recommendation.cardinality if recommendation else "UNKNOWN"
        join_type: RecommendedJoinType = recommendation.recommended_join if recommendation else "NO_SAFE_JOIN"
        confidence = recommendation.confidence if recommendation else 0

        result = JoinExecutionResult(
            status="BLOCKED",
            safe_to_execute=False,
            left_dataset=name_a,
            right_dataset=name_b,
            left_column=col_left,
            right_column=col_right,
            join_type=join_type,
            cardinality=cardinality,
            confidence=confidence,
            rows_before_left=0,
            rows_before_right=0,
            rows_after=0,
            columns_before=0,
            columns_after=0,
            matched_rows=0,
            unmatched_left_rows=0,
            unmatched_right_rows=0,
            row_multiplication_factor=0.0,
            duplicate_expansion_detected=False,
            derived_dataset_name=derived_name,
            warnings=[f"Execution blocked: {error_reason}"],
            evidence=["Safety verification failed before pandas.merge(). Merge was NOT called."],
            error_reason=error_reason,
            execution_timestamp=now_ts,
        )

        return SafeJoinExecutionOutput(result=result, dataframe=None)
