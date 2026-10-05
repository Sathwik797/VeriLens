import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Set, Tuple

import pandas as pd

from app.eda.column_normalizer import ColumnNormalizer
from app.eda.dataset_relationship import DatasetCompatibilityAnalyzer
from app.models.dataset_profile import DatasetProfile
from app.models.dataset_relationship import DatasetRelationship
from app.models.join_recommendation import (
    CandidateJoinKey,
    ConfidenceLevel,
    JoinCardinality,
    JoinRecommendation,
    RecommendedJoinType,
)

logger = logging.getLogger(__name__)


class JoinIntelligenceAnalyzer:
    """
    Deterministic Safe Join Intelligence Analyzer.
    Evaluates candidate join keys between datasets, determines cardinality,
    measures key coverage and unmatched records, and produces a safe join
    recommendation without executing any merge.
    """

    @classmethod
    def calculate_confidence(
        cls,
        type_strength: str,
        coverage_pct: float,
        cardinality: JoinCardinality,
        name_alignment_pts: int,
        left_null_pct: float,
        right_null_pct: float,
        is_safe: bool,
    ) -> Tuple[int, ConfidenceLevel]:
        """
        Deterministic, bounded confidence score formula (0 - 100):
        1. Type Compatibility (up to 25 pts)
        2. Value Coverage (up to 35 pts)
        3. Cardinality & Uniqueness (up to 25 pts)
        4. Name Alignment (up to 15 pts)
        5. Penalties for nulls, low coverage, many-to-many, or unsafe conditions.
        """
        score = 0.0

        # 1. Type compatibility
        if type_strength == "STRONG":
            score += 25.0
        elif type_strength == "COMPATIBLE":
            score += 15.0
        else:
            score += 0.0

        # 2. Value coverage
        score += (max(0.0, min(100.0, coverage_pct)) / 100.0) * 35.0

        # 3. Cardinality
        if cardinality in ("ONE_TO_ONE", "ONE_TO_MANY", "MANY_TO_ONE"):
            score += 25.0
        elif cardinality == "MANY_TO_MANY":
            score += 5.0
        else:
            score += 0.0

        # 4. Name alignment
        score += float(name_alignment_pts)

        # 5. Penalties
        avg_null = (left_null_pct + right_null_pct) / 2.0
        if avg_null > 0:
            score -= min(15.0, avg_null * 0.5)

        if cardinality == "MANY_TO_MANY":
            score -= 20.0

        if coverage_pct < 50.0:
            score -= 25.0

        if not is_safe:
            score = min(score, 45.0)

        # Clamp between 0 and 100
        final_score = int(round(max(0.0, min(100.0, score))))

        if final_score >= 90:
            level: ConfidenceLevel = "HIGH"
        elif final_score >= 70:
            level = "MODERATE"
        elif final_score >= 50:
            level = "LOW"
        else:
            level = "VERY LOW"

        return final_score, level

    @classmethod
    def evaluate_type_strength(cls, dtype_a: str, dtype_b: str) -> str:
        """Determines type compatibility strength."""
        if not DatasetCompatibilityAnalyzer.are_types_compatible(dtype_a, dtype_b):
            return "INCOMPATIBLE"

        # Check strong matching families
        int_types = {"int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
        str_types = {"object", "string", "str"}
        dt_types = {"datetime64[ns]", "datetime64", "datetime"}

        a_lower = dtype_a.lower()
        b_lower = dtype_b.lower()

        if any(t in a_lower for t in int_types) and any(t in b_lower for t in int_types):
            return "STRONG"
        if any(t in a_lower for t in str_types) and any(t in b_lower for t in str_types):
            return "STRONG"
        if any(t in a_lower for t in dt_types) and any(t in b_lower for t in dt_types):
            return "STRONG"

        return "COMPATIBLE"

    @classmethod
    def discover_candidate_keys(
        cls,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        profile_a: Optional[DatasetProfile] = None,
        profile_b: Optional[DatasetProfile] = None,
    ) -> List[Tuple[str, str, int]]:
        """
        Discovers potential join key pairs between df_a and df_b.
        Returns a list of tuples: (col_a, col_b, name_alignment_pts).
        Filters out index-like columns and ranks by name similarity.
        """
        candidate_pairs: List[Tuple[str, str, int]] = []
        seen_pairs: Set[Tuple[str, str]] = set()

        cols_a = [c for c in df_a.columns if not ColumnNormalizer.is_index_like(str(c), df_a[c])]
        cols_b = [c for c in df_b.columns if not ColumnNormalizer.is_index_like(str(c), df_b[c])]

        norm_map_a = {str(c): ColumnNormalizer.normalize(str(c)) for c in cols_a}
        norm_map_b = {str(c): ColumnNormalizer.normalize(str(c)) for c in cols_b}

        for c_a in cols_a:
            str_a = str(c_a)
            norm_a = norm_map_a[str_a]
            is_biz_a = ColumnNormalizer.is_business_identifier(str_a)

            for c_b in cols_b:
                str_b = str(c_b)
                pair_key = (str_a, str_b)
                if pair_key in seen_pairs:
                    continue

                norm_b = norm_map_b[str_b]
                is_biz_b = ColumnNormalizer.is_business_identifier(str_b)

                name_pts = 0
                # 1. Exact case-insensitive match
                if str_a.strip().lower() == str_b.strip().lower():
                    name_pts = 15
                # 2. Normalized stem match (e.g. cust_id vs customer_id)
                elif norm_a and norm_a == norm_b:
                    name_pts = 13
                # 3. Both are business identifiers and share entity prefix
                elif is_biz_a and is_biz_b and ColumnNormalizer.similarity(str_a, str_b) >= 0.5:
                    name_pts = 12
                # 4. Token Jaccard similarity >= 0.60
                elif ColumnNormalizer.similarity(str_a, str_b) >= 0.60:
                    name_pts = 10
                # 5. Check if one is a candidate key in profile
                elif (
                    profile_a
                    and profile_b
                    and (str_a in profile_a.candidate_keys or str_b in profile_b.candidate_keys)
                    and ColumnNormalizer.similarity(str_a, str_b) >= 0.40
                ):
                    name_pts = 8

                if name_pts > 0:
                    candidate_pairs.append((str_a, str_b, name_pts))
                    seen_pairs.add(pair_key)

        return candidate_pairs

    @classmethod
    def evaluate_key_pair(
        cls,
        col_a: str,
        df_a: pd.DataFrame,
        name_a: str,
        col_b: str,
        df_b: pd.DataFrame,
        name_b: str,
        name_alignment_pts: int,
    ) -> CandidateJoinKey:
        """
        Calculates deterministic validation metrics for a single candidate key pair.
        """
        n_a = len(df_a)
        n_b = len(df_b)

        dtype_a = str(df_a[col_a].dtype)
        dtype_b = str(df_b[col_b].dtype)
        type_strength = cls.evaluate_type_strength(dtype_a, dtype_b)
        type_compatible = type_strength != "INCOMPATIBLE"

        # Null metrics
        null_a = int(df_a[col_a].isnull().sum())
        null_b = int(df_b[col_b].isnull().sum())
        null_pct_a = round((null_a / n_a * 100), 2) if n_a > 0 else 0.0
        null_pct_b = round((null_b / n_b * 100), 2) if n_b > 0 else 0.0

        # Uniqueness metrics
        u_a = int(df_a[col_a].nunique())
        u_b = int(df_b[col_b].nunique())
        left_unique = (u_a == n_a) and (null_a == 0) and (n_a > 0)
        right_unique = (u_b == n_b) and (null_b == 0) and (n_b > 0)

        # Cardinality
        if left_unique and right_unique:
            cardinality: JoinCardinality = "ONE_TO_ONE"
        elif left_unique and not right_unique:
            cardinality = "ONE_TO_MANY"
        elif not left_unique and right_unique:
            cardinality = "MANY_TO_ONE"
        elif not left_unique and not right_unique:
            cardinality = "MANY_TO_MANY"
        else:
            cardinality = "UNKNOWN"

        # Parent / Child direction
        parent_dataset: Optional[str] = None
        parent_key: Optional[str] = None
        child_dataset: Optional[str] = None
        child_key: Optional[str] = None

        if cardinality == "ONE_TO_MANY":
            parent_dataset = name_a
            parent_key = col_a
            child_dataset = name_b
            child_key = col_b
        elif cardinality == "MANY_TO_ONE":
            parent_dataset = name_b
            parent_key = col_b
            child_dataset = name_a
            child_key = col_a
        elif cardinality == "ONE_TO_ONE":
            # Default parent to left side for one-to-one
            parent_dataset = name_a
            parent_key = col_a
            child_dataset = name_b
            child_key = col_b

        # Value Coverage via Set-based operations
        s_a = df_a[col_a].dropna().astype(str).str.strip()
        s_b = df_b[col_b].dropna().astype(str).str.strip()

        distinct_a = set(s_a.unique())
        distinct_b = set(s_b.unique())
        common_keys = distinct_a.intersection(distinct_b)

        # Matched and unmatched row counts
        # Rows in A that have a matching key in B
        matched_in_a = int(s_a.isin(distinct_b).sum()) if distinct_b else 0
        unmatched_left = n_a - matched_in_a

        # Rows in B that have a matching key in A
        matched_in_b = int(s_b.isin(distinct_a).sum()) if distinct_a else 0
        unmatched_right = n_b - matched_in_b

        # Coverage percentage:
        # In a parent-child relationship (e.g. Customers -> Orders),
        # coverage represents the fraction of child records that match a parent record.
        if cardinality == "ONE_TO_MANY" and n_b > 0:
            coverage_pct = round((matched_in_b / n_b * 100.0), 2)
            matched_rows = matched_in_b
        elif cardinality == "MANY_TO_ONE" and n_a > 0:
            coverage_pct = round((matched_in_a / n_a * 100.0), 2)
            matched_rows = matched_in_a
        elif cardinality == "ONE_TO_ONE":
            denom = max(n_a, n_b, 1)
            matched_rows = len(common_keys)
            coverage_pct = round((len(common_keys) / max(len(distinct_a.union(distinct_b)), 1) * 100.0), 2)
        else:
            # Many-to-many
            denom = max(n_a, n_b, 1)
            matched_rows = min(matched_in_a, matched_in_b)
            coverage_pct = round((min(matched_in_a, matched_in_b) / denom * 100.0), 2)

        # -------------------------------------------------------------
        # Safety Evaluation
        # -------------------------------------------------------------
        warnings: List[str] = []
        evidence: List[str] = []
        is_safe = True

        if not type_compatible:
            is_safe = False
            warnings.append(
                f"Incompatible data types: '{col_a}' ({dtype_a}) and '{col_b}' ({dtype_b})."
            )

        if ColumnNormalizer.is_index_like(col_a, df_a[col_a]) or ColumnNormalizer.is_index_like(col_b, df_b[col_b]):
            is_safe = False
            warnings.append(
                f"Candidate key '{col_a}' or '{col_b}' is an index-like or row counter column."
            )

        if null_pct_a > 10.0 or null_pct_b > 10.0:
            is_safe = False
            warnings.append(
                f"Excessive null values: {null_pct_a}% in '{name_a}.{col_a}', {null_pct_b}% in '{name_b}.{col_b}'."
            )
        elif null_a > 0 or null_b > 0:
            warnings.append(
                f"Null values present: {null_a} rows ({null_pct_a}%) in {name_a}, {null_b} rows ({null_pct_b}%) in {name_b}."
            )

        if cardinality == "MANY_TO_MANY":
            is_safe = False
            warnings.append(
                f"Many-to-many relationship detected on '{col_a}' <-> '{col_b}'. "
                "Joining without a unique primary key risks combinatorial row duplication (Cartesian explosion)."
            )

        if coverage_pct < 50.0:
            is_safe = False
            warnings.append(
                f"Low key value coverage ({coverage_pct}%). Less than 50% of records find a match."
            )

        if len(common_keys) == 0:
            is_safe = False
            warnings.append("No overlapping values found between candidate keys.")

        # Unmatched warnings
        if cardinality == "ONE_TO_MANY" and unmatched_right > 0:
            unmatched_pct = round((unmatched_right / n_b * 100), 2)
            warnings.append(
                f"{unmatched_pct}% of records in '{name_b}' ({unmatched_right:,} rows) have no matching parent in '{name_a}'."
            )
        elif cardinality == "MANY_TO_ONE" and unmatched_left > 0:
            unmatched_pct = round((unmatched_left / n_a * 100), 2)
            warnings.append(
                f"{unmatched_pct}% of records in '{name_a}' ({unmatched_left:,} rows) have no matching parent in '{name_b}'."
            )

        # Evidence generation
        if left_unique:
            evidence.append(f"'{col_a}' has unique values (zero duplicates, zero nulls) in '{name_a}'.")
        else:
            evidence.append(f"'{col_a}' contains recurring values in '{name_a}' ({u_a:,} unique in {n_a:,} rows).")

        if right_unique:
            evidence.append(f"'{col_b}' has unique values (zero duplicates, zero nulls) in '{name_b}'.")
        else:
            evidence.append(f"'{col_b}' contains recurring values in '{name_b}' ({u_b:,} unique in {n_b:,} rows).")

        evidence.append(f"Key value coverage: {coverage_pct}% matching records ({len(common_keys):,} shared distinct keys).")

        if type_strength == "STRONG":
            evidence.append(f"Strong data type compatibility: {dtype_a} <-> {dtype_b}.")
        elif type_strength == "COMPATIBLE":
            evidence.append(f"Compatible data types: {dtype_a} <-> {dtype_b}.")

        # Join Recommendation
        if not is_safe:
            recommended_join: RecommendedJoinType = "NO_SAFE_JOIN"
        else:
            # Safe join logic
            if cardinality == "ONE_TO_ONE" and coverage_pct >= 99.0 and unmatched_left == 0 and unmatched_right == 0:
                recommended_join = "INNER_JOIN"
            elif cardinality in ("ONE_TO_MANY", "MANY_TO_ONE", "ONE_TO_ONE"):
                recommended_join = "LEFT_JOIN"
            else:
                recommended_join = "NO_SAFE_JOIN"

        # Deterministic confidence
        confidence, confidence_level = cls.calculate_confidence(
            type_strength=type_strength,
            coverage_pct=coverage_pct,
            cardinality=cardinality,
            name_alignment_pts=name_alignment_pts,
            left_null_pct=null_pct_a,
            right_null_pct=null_pct_b,
            is_safe=is_safe,
        )

        return CandidateJoinKey(
            left_column=col_a,
            right_column=col_b,
            left_dtype=dtype_a,
            right_dtype=dtype_b,
            left_unique=left_unique,
            right_unique=right_unique,
            left_null_count=null_a,
            right_null_count=null_b,
            left_null_percentage=null_pct_a,
            right_null_percentage=null_pct_b,
            matched_rows=matched_rows,
            unmatched_left_rows=unmatched_left,
            unmatched_right_rows=unmatched_right,
            coverage_percentage=coverage_pct,
            cardinality=cardinality,
            confidence=confidence,
            confidence_level=confidence_level,
            recommended_join=recommended_join,
            parent_dataset=parent_dataset,
            parent_key=parent_key,
            child_dataset=child_dataset,
            child_key=child_key,
            safe_to_execute=is_safe,
            evidence=evidence,
            warnings=warnings,
        )

    @classmethod
    def analyze_pair(
        cls,
        profile_a: DatasetProfile,
        df_a: pd.DataFrame,
        profile_b: DatasetProfile,
        df_b: pd.DataFrame,
        relationship: Optional[DatasetRelationship] = None,
        use_gemini: bool = True,
    ) -> JoinRecommendation:
        """
        Performs safe join intelligence analysis between dataset A and dataset B.
        Only produces join recommendations; NEVER executes pandas.merge() or joins.
        """
        name_a = profile_a.filename or "dataset_a"
        name_b = profile_b.filename or "dataset_b"

        # If relationship is provided and is UNRELATED or COMPATIBLE (not relational),
        # return a safe rejection unless explicitly testing
        if relationship and relationship.classification != "RELATED":
            return JoinRecommendation(
                dataset_a=name_a,
                dataset_b=name_b,
                left_column="",
                right_column="",
                relationship_type=relationship.classification,
                cardinality="UNKNOWN",
                confidence=0,
                confidence_level="VERY LOW",
                left_unique=False,
                right_unique=False,
                left_null_percentage=0.0,
                right_null_percentage=0.0,
                matched_rows=0,
                unmatched_left_rows=len(df_a),
                unmatched_right_rows=len(df_b),
                coverage_percentage=0.0,
                recommended_join="NO_SAFE_JOIN",
                safe_to_execute=False,
                evidence=["Datasets are not classified as RELATED entity tables."],
                warnings=[f"Relationship classification is {relationship.classification}. No safe join recommended."],
                explanation=f"Datasets '{name_a}' and '{name_b}' do not represent relational entities.",
                all_candidates=[],
            )

        # 1. Discover candidate key pairs
        candidate_pairs = cls.discover_candidate_keys(df_a, df_b, profile_a, profile_b)

        if not candidate_pairs:
            return JoinRecommendation(
                dataset_a=name_a,
                dataset_b=name_b,
                left_column="",
                right_column="",
                relationship_type="RELATED",
                cardinality="UNKNOWN",
                confidence=0,
                confidence_level="VERY LOW",
                left_unique=False,
                right_unique=False,
                left_null_percentage=0.0,
                right_null_percentage=0.0,
                matched_rows=0,
                unmatched_left_rows=len(df_a),
                unmatched_right_rows=len(df_b),
                coverage_percentage=0.0,
                recommended_join="NO_SAFE_JOIN",
                safe_to_execute=False,
                evidence=["No valid candidate join keys detected."],
                warnings=["No matching or semantically compatible join keys found."],
                explanation="No safe join key candidate could be identified between the datasets.",
                all_candidates=[],
            )

        # 2. Evaluate each candidate key pair
        evaluated_candidates: List[CandidateJoinKey] = []
        for col_a, col_b, name_pts in candidate_pairs:
            cand = cls.evaluate_key_pair(
                col_a=col_a,
                df_a=df_a,
                name_a=name_a,
                col_b=col_b,
                df_b=df_b,
                name_b=name_b,
                name_alignment_pts=name_pts,
            )
            evaluated_candidates.append(cand)

        # 3. Sort candidates by deterministic confidence (descending)
        evaluated_candidates.sort(
            key=lambda c: (c.safe_to_execute, c.confidence, c.coverage_percentage),
            reverse=True,
        )

        top_candidate = evaluated_candidates[0]

        # Check for ambiguity: multiple top candidates with identical high confidence
        extra_warnings = list(top_candidate.warnings)
        if len(evaluated_candidates) > 1:
            cand_2 = evaluated_candidates[1]
            if cand_2.safe_to_execute and (top_candidate.confidence - cand_2.confidence < 5):
                extra_warnings.append(
                    f"Multiple viable join keys detected with close confidence: '{top_candidate.left_column}' <-> '{top_candidate.right_column}' ({top_candidate.confidence}%) vs '{cand_2.left_column}' <-> '{cand_2.right_column}' ({cand_2.confidence}%). Manual review recommended."
                )

        # Deterministic explanation
        explanation = cls._generate_explanation(top_candidate, name_a, name_b)

        # Optional Gemini semantic enrichment
        if use_gemini and top_candidate.safe_to_execute:
            explanation = cls._maybe_enhance_explanation_with_gemini(
                top_candidate, name_a, name_b, default_explanation=explanation
            )

        return JoinRecommendation(
            dataset_a=name_a,
            dataset_b=name_b,
            left_column=top_candidate.left_column,
            right_column=top_candidate.right_column,
            relationship_type="RELATED",
            cardinality=top_candidate.cardinality,
            confidence=top_candidate.confidence,
            confidence_level=top_candidate.confidence_level,
            left_unique=top_candidate.left_unique,
            right_unique=top_candidate.right_unique,
            left_null_percentage=top_candidate.left_null_percentage,
            right_null_percentage=top_candidate.right_null_percentage,
            matched_rows=top_candidate.matched_rows,
            unmatched_left_rows=top_candidate.unmatched_left_rows,
            unmatched_right_rows=top_candidate.unmatched_right_rows,
            coverage_percentage=top_candidate.coverage_percentage,
            recommended_join=top_candidate.recommended_join,
            parent_dataset=top_candidate.parent_dataset,
            parent_key=top_candidate.parent_key,
            child_dataset=top_candidate.child_dataset,
            child_key=top_candidate.child_key,
            safe_to_execute=top_candidate.safe_to_execute,
            evidence=top_candidate.evidence,
            warnings=extra_warnings,
            explanation=explanation,
            all_candidates=evaluated_candidates,
        )

    @classmethod
    def _generate_explanation(cls, cand: CandidateJoinKey, name_a: str, name_b: str) -> str:
        """Generates clear, factual deterministic explanation."""
        if not cand.safe_to_execute:
            reason = cand.warnings[0] if cand.warnings else "Safety constraints violated."
            return f"No safe join recommended between '{name_a}' and '{name_b}'. Reason: {reason}"

        card_str = cand.cardinality.lower().replace("_", "-")
        if cand.parent_dataset and cand.child_dataset:
            return (
                f"'{cand.parent_key}' in '{cand.parent_dataset}' acts as candidate primary key "
                f"linking to '{cand.child_key}' in '{cand.child_dataset}' ({card_str} relationship) "
                f"with {cand.coverage_percentage}% key coverage. Preserves transactional records while enriching entity data."
            )
        return (
            f"Candidate join '{cand.left_column}' <-> '{cand.right_column}' forms a {card_str} relationship "
            f"with {cand.coverage_percentage}% key coverage."
        )

    @classmethod
    def _maybe_enhance_explanation_with_gemini(
        cls,
        cand: CandidateJoinKey,
        name_a: str,
        name_b: str,
        default_explanation: str,
    ) -> str:
        """
        Optionally uses Gemini to refine the natural-language explanation.
        NEVER allows Gemini to calculate or override metrics, confidence, or safety.
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return default_explanation

        try:
            import google.generativeai as genai

            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(
                model_name="gemini-2.5-flash",
                generation_config={"temperature": 0.1, "response_mime_type": "application/json"},
            )

            prompt = f"""
You are a database and data integrity expert.
Provide a concise, professional 1-2 sentence explanation for a proposed join between two tables.
Do NOT change any numbers or facts. Only output JSON matching:
{{"explanation": "your explanation"}}

Metadata:
- Table A: {name_a}
- Table B: {name_b}
- Left Key: {cand.left_column} ({cand.left_dtype})
- Right Key: {cand.right_column} ({cand.right_dtype})
- Cardinality: {cand.cardinality}
- Key Coverage: {cand.coverage_percentage}%
- Recommended Join: {cand.recommended_join}
- Parent: {cand.parent_dataset} ({cand.parent_key})
- Child: {cand.child_dataset} ({cand.child_key})
"""
            response = model.generate_content(prompt)
            text = response.text.strip()
            # Clean markdown codeblocks if any
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?", "", text)
                text = re.sub(r"```$", "", text).strip()

            data = json.loads(text)
            if "explanation" in data and len(data["explanation"]) > 10:
                return data["explanation"].strip()

        except Exception as e:
            logger.info(f"Gemini join explanation skipped ({e}). Using deterministic explanation.")

        return default_explanation
