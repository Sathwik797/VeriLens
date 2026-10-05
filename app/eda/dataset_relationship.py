import json
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple
import pandas as pd

from app.eda.column_normalizer import ColumnNormalizer
from app.eda.loader import DataLoader
from app.eda.profiler import DataProfiler
from app.models.dataset_profile import DatasetProfile
from app.models.dataset_relationship import DatasetRelationship, RelationshipType

logger = logging.getLogger(__name__)


class DatasetCompatibilityAnalyzer:
    """
    Deterministic compatibility and relationship engine for multi-dataset workspaces.
    Evaluates schema overlap, type compatibility, candidate keys, cardinality,
    and value overlap to classify pairs into COMPATIBLE, RELATED, or UNRELATED.
    """

    @classmethod
    def are_types_compatible(cls, type_a: str, type_b: str) -> bool:
        """Determines if two pandas data types are semantically compatible."""
        ta = type_a.lower()
        tb = type_b.lower()

        # Both numeric
        num_patterns = ["int", "float", "double", "num"]
        if any(p in ta for p in num_patterns) and any(p in tb for p in num_patterns):
            return True

        # Both text/category
        cat_patterns = ["object", "str", "category"]
        if any(p in ta for p in cat_patterns) and any(p in tb for p in cat_patterns):
            return True

        # Both date/time
        date_patterns = ["datetime", "timestamp", "date"]
        if any(p in ta for p in date_patterns) and any(p in tb for p in date_patterns):
            return True

        # Both bool
        if "bool" in ta and "bool" in tb:
            return True

        return False

    @classmethod
    def compute_value_overlap(
        cls, series_a: pd.Series, series_b: pd.Series, max_samples: int = 500
    ) -> Tuple[float, int]:
        """
        Efficiently samples unique values from both series to compute set overlap.
        Avoids full dataframe cartesian products.
        """
        vals_a = series_a.dropna()
        vals_b = series_b.dropna()

        if vals_a.empty or vals_b.empty:
            return 0.0, 0

        # Sample unique values for performance
        u_a = vals_a.astype(str).str.strip().str.lower().unique()
        u_b = vals_b.astype(str).str.strip().str.lower().unique()

        if len(u_a) > max_samples:
            u_a = pd.Series(u_a).sample(max_samples, random_state=42).values
        if len(u_b) > max_samples:
            u_b = pd.Series(u_b).sample(max_samples, random_state=42).values

        set_a = set(u_a)
        set_b = set(u_b)

        intersection = set_a.intersection(set_b)
        n_common = len(intersection)
        min_card = min(len(set_a), len(set_b))

        overlap_ratio = (n_common / min_card) if min_card > 0 else 0.0
        return overlap_ratio, n_common

    @classmethod
    def analyze_pair(
        cls,
        profile_a: DatasetProfile,
        df_a: pd.DataFrame,
        profile_b: DatasetProfile,
        df_b: pd.DataFrame,
        use_gemini: bool = True,
    ) -> DatasetRelationship:
        """
        Analyzes relationship between dataset A and dataset B using deterministic
        signals, with optional Gemini semantic enhancement.
        """
        name_a = profile_a.filename or "dataset_a"
        name_b = profile_b.filename or "dataset_b"

        cols_a = profile_a.column_names
        cols_b = profile_b.column_names

        # Map normalized column -> original column
        norm_map_a = {v: k for k, v in profile_a.normalized_columns.items()}
        norm_map_b = {v: k for k, v in profile_b.normalized_columns.items()}

        set_norm_a = set(norm_map_a.keys())
        set_norm_b = set(norm_map_b.keys())

        shared_norm = sorted(list(set_norm_a.intersection(set_norm_b)))
        total_union = set_norm_a.union(set_norm_b)

        jaccard = (len(shared_norm) / len(total_union)) if total_union else 0.0
        min_cols = min(len(cols_a), len(cols_b)) if (cols_a and cols_b) else 1
        max_cols = max(len(cols_a), len(cols_b)) if (cols_a and cols_b) else 1
        size_ratio = min_cols / max_cols

        # Evaluate type compatibility of shared columns
        compatible_type_cols = []
        incompatible_type_cols = []

        for norm_col in shared_norm:
            orig_a = norm_map_a[norm_col]
            orig_b = norm_map_b[norm_col]
            t_a = profile_a.data_types.get(orig_a, "")
            t_b = profile_b.data_types.get(orig_b, "")

            if cls.are_types_compatible(t_a, t_b):
                compatible_type_cols.append(orig_a)
            else:
                incompatible_type_cols.append((orig_a, orig_b))

        type_compat_ratio = (
            len(compatible_type_cols) / len(shared_norm) if shared_norm else 0.0
        )

        # -------------------------------------------------------------
        # Signal 1: Check for COMPATIBLE (Schema Homogeneity)
        # -------------------------------------------------------------
        # Criteria: High normalized column overlap (>=70%) AND similar column count
        # AND high type compatibility.
        if (jaccard >= 0.70 or (len(shared_norm) / min_cols >= 0.80 and size_ratio >= 0.70)) and type_compat_ratio >= 0.75:
            classification: RelationshipType = "COMPATIBLE"
            conf = min(0.98, max(0.80, jaccard * 0.6 + type_compat_ratio * 0.4))
            evidence = [
                f"High schema similarity: {int(jaccard * 100)}% normalized column overlap ({len(shared_norm)} matching columns).",
                f"Data type compatibility: {int(type_compat_ratio * 100)}% across shared schema.",
                f"Column count parity: {len(cols_a)} columns in '{name_a}' vs {len(cols_b)} columns in '{name_b}'.",
            ]
            recommendation = (
                "Datasets share an equivalent structure and are suitable for vertical concatenation (union/append)."
            )
            rel = DatasetRelationship(
                dataset_a=name_a,
                dataset_b=name_b,
                classification=classification,
                confidence=round(conf, 2),
                shared_columns=[norm_map_a[c] for c in shared_norm],
                candidate_keys=[],
                evidence=evidence,
                recommendation=recommendation,
            )
            return cls._maybe_enhance_with_gemini(rel, profile_a, profile_b) if use_gemini else rel

        # -------------------------------------------------------------
        # Signal 2: Check for RELATED (Entity Linkage / Foreign Keys)
        # -------------------------------------------------------------
        # Inspect shared columns that might act as entity identifiers / keys
        candidate_keys_a_norm = {profile_a.normalized_columns.get(c, "") for c in profile_a.candidate_keys}
        candidate_keys_b_norm = {profile_b.normalized_columns.get(c, "") for c in profile_b.candidate_keys}

        potential_link_keys = []
        key_evidence = []

        for norm_col in shared_norm:
            orig_a = norm_map_a[norm_col]
            orig_b = norm_map_b[norm_col]

            # Index-like columns (row numbers, indices, Unnamed counters) must NEVER be linking keys
            if ColumnNormalizer.is_index_like(orig_a, df_a[orig_a]) or ColumnNormalizer.is_index_like(orig_b, df_b[orig_b]):
                continue

            # Check if types are compatible
            t_a = profile_a.data_types.get(orig_a, "")
            t_b = profile_b.data_types.get(orig_b, "")
            if not cls.are_types_compatible(t_a, t_b):
                continue

            is_key_a = norm_col in candidate_keys_a_norm
            is_key_b = norm_col in candidate_keys_b_norm
            is_id_name = (
                any(k in norm_col for k in ["id", "code", "key", "number", "num"])
                or ColumnNormalizer.is_business_identifier(orig_a)
                or ColumnNormalizer.is_business_identifier(orig_b)
            )

            if is_key_a or is_key_b or is_id_name:
                # Measure value overlap
                overlap_ratio, n_common = cls.compute_value_overlap(df_a[orig_a], df_b[orig_b])

                # A valid relational link must either have explicit ID naming, or have positive value overlap
                if not is_id_name and n_common == 0:
                    continue

                # Determine cardinality pattern
                u_a = profile_a.unique_counts.get(orig_a, df_a[orig_a].nunique())
                u_b = profile_b.unique_counts.get(orig_b, df_b[orig_b].nunique())
                r_a = (u_a / profile_a.rows) if profile_a.rows > 0 else 0
                r_b = (u_b / profile_b.rows) if profile_b.rows > 0 else 0

                if r_a >= 0.95 and r_b < 0.90:
                    cardinality = f"1-to-many (Primary in {name_a}, Foreign in {name_b})"
                elif r_b >= 0.95 and r_a < 0.90:
                    cardinality = f"1-to-many (Primary in {name_b}, Foreign in {name_a})"
                elif r_a >= 0.95 and r_b >= 0.95:
                    cardinality = "1-to-1 key relationship"
                else:
                    cardinality = "many-to-many entity relationship"

                potential_link_keys.append(norm_col)
                ev = f"Potential key '{orig_a}' ↔ '{orig_b}': {cardinality}."
                if n_common > 0:
                    ev += f" Value overlap: {int(overlap_ratio * 100)}% ({n_common} matching entities)."
                key_evidence.append(ev)

        if potential_link_keys:
            classification = "RELATED"
            conf = 0.85 if any("Value overlap" in ev for ev in key_evidence) else 0.75
            evidence = [
                f"Datasets represent distinct entities with shared linking columns: {', '.join(potential_link_keys)}.",
                *key_evidence,
            ]
            primary_key = potential_link_keys[0]
            recommendation = (
                f"Datasets appear relational. Suitable for joining or relational querying on key: '{norm_map_a.get(primary_key, primary_key)}'."
            )
            rel = DatasetRelationship(
                dataset_a=name_a,
                dataset_b=name_b,
                classification=classification,
                confidence=round(conf, 2),
                shared_columns=[norm_map_a[c] for c in shared_norm],
                candidate_keys=potential_link_keys,
                evidence=evidence,
                recommendation=recommendation,
            )
            return cls._maybe_enhance_with_gemini(rel, profile_a, profile_b) if use_gemini else rel

        # -------------------------------------------------------------
        # Signal 3: Default to UNRELATED
        # -------------------------------------------------------------
        classification = "UNRELATED"
        conf = round(max(0.80, 1.0 - jaccard), 2)
        evidence = [
            f"Low schema overlap: only {len(shared_norm)} shared normalized columns ({int(jaccard * 100)}% overlap).",
            "No candidate foreign or primary key relationships detected.",
            "Insufficient evidence of structural or relational linkage.",
        ]
        if incompatible_type_cols:
            evidence.append(
                f"Type mismatches detected on shared names: {', '.join(f'{a}<->{b}' for a, b in incompatible_type_cols[:3])}."
            )

        recommendation = (
            "Datasets appear semantically and structurally independent. Recommend analyzing as separate tables."
        )

        rel = DatasetRelationship(
            dataset_a=name_a,
            dataset_b=name_b,
            classification=classification,
            confidence=conf,
            shared_columns=[norm_map_a[c] for c in shared_norm],
            candidate_keys=[],
            evidence=evidence,
            recommendation=recommendation,
        )
        return cls._maybe_enhance_with_gemini(rel, profile_a, profile_b) if use_gemini else rel

    @classmethod
    def _maybe_enhance_with_gemini(
        cls,
        deterministic_rel: DatasetRelationship,
        profile_a: DatasetProfile,
        profile_b: DatasetProfile,
    ) -> DatasetRelationship:
        """
        Optionally uses Gemini to refine recommendations or verify relationship.
        Strictly falls back to deterministic result if Gemini is offline, rate-limited,
        times out, or returns invalid structure.
        NEVER sends raw dataframe rows or values!
        """
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return deterministic_rel

        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            llm = genai.GenerativeModel(
                model_name=os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )

            # Compact metadata only (NO raw dataframe rows!)
            compact_payload = {
                "dataset_a": {
                    "name": profile_a.filename or "dataset_a",
                    "columns": profile_a.column_names,
                    "data_types": profile_a.data_types,
                },
                "dataset_b": {
                    "name": profile_b.filename or "dataset_b",
                    "columns": profile_b.column_names,
                    "data_types": profile_b.data_types,
                },
                "deterministic_signals": {
                    "classification": deterministic_rel.classification,
                    "confidence": deterministic_rel.confidence,
                    "shared_columns": deterministic_rel.shared_columns,
                    "candidate_keys": deterministic_rel.candidate_keys,
                    "evidence": deterministic_rel.evidence,
                },
            }

            prompt = f"""You are the schema compatibility reasoning component of VeriLens AI.
Evaluate whether dataset_a and dataset_b are COMPATIBLE, RELATED, or UNRELATED based strictly on the metadata provided.

Definitions:
- COMPATIBLE: Highly similar schemas suitable for appending or unioning.
- RELATED: Distinct schemas representing related entities with foreign/primary key linkage (e.g. customers and orders).
- UNRELATED: Independent datasets with no clear relational or schema connection.

Input Metadata:
{json.dumps(compact_payload, indent=2)}

Respond ONLY with valid JSON in this exact structure:
{{
  "classification": "COMPATIBLE" | "RELATED" | "UNRELATED",
  "confidence": <float between 0.5 and 1.0>,
  "reasoning": "<1-2 sentence explanation>",
  "recommendation": "<prescriptive guidance for user>"
}}
"""
            response = llm.generate_content(
                prompt,
                generation_config=dict(temperature=0.0, response_mime_type="application/json"),
            )
            data = json.loads(response.text.strip())

            cls_candidate = str(data.get("classification", "")).upper()
            if cls_candidate in ["COMPATIBLE", "RELATED", "UNRELATED"]:
                new_ev = list(deterministic_rel.evidence)
                if data.get("reasoning"):
                    new_ev.append(f"Semantic reasoning: {data['reasoning']}")

                return DatasetRelationship(
                    dataset_a=deterministic_rel.dataset_a,
                    dataset_b=deterministic_rel.dataset_b,
                    classification=cls_candidate,  # type: ignore
                    confidence=float(data.get("confidence", deterministic_rel.confidence)),
                    shared_columns=deterministic_rel.shared_columns,
                    candidate_keys=deterministic_rel.candidate_keys,
                    evidence=new_ev,
                    recommendation=data.get("recommendation", deterministic_rel.recommendation),
                )

        except Exception as e:
            logger.info(f"Gemini relationship enhancement skipped ({e}). Using deterministic result.")

        return deterministic_rel

    @classmethod
    def analyze_workspace(
        cls,
        files: List[Any],
        use_gemini: bool = True,
    ) -> Tuple[List[DatasetProfile], List[DatasetRelationship]]:
        """
        Loads all files in the workspace once, profiles each independently,
        and analyzes compatibility for all dataset pairs.
        """
        profiles: List[DatasetProfile] = []
        dfs: List[pd.DataFrame] = []

        if not files:
            return [], []

        # 1. Ingest & Profile each file independently
        for file_obj in files:
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            orig_name = getattr(file_obj, "orig_name", None) or os.path.basename(path)

            try:
                df = DataLoader.load_file(path)
                df.attrs["filename"] = orig_name
                profile = DataProfiler.generate_profile(df, filename=orig_name)
                profiles.append(profile)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to load workspace file {orig_name}: {e}")

        # 2. Pairwise compatibility analysis
        relationships: List[DatasetRelationship] = []
        n = len(profiles)

        for i in range(n):
            for j in range(i + 1, n):
                rel = cls.analyze_pair(
                    profiles[i], dfs[i], profiles[j], dfs[j], use_gemini=use_gemini
                )
                relationships.append(rel)

        return profiles, relationships
