import unittest
from unittest.mock import patch
import os
import tempfile
import pandas as pd

from app.eda.safe_join_executor import SafeJoinExecutor
from app.models.join_recommendation import JoinRecommendation, CandidateJoinKey
from app.models.join_execution_result import JoinExecutionResult
from app.eda.profiler import DataProfiler
from app.eda.join_intelligence import JoinIntelligenceAnalyzer
from app.services.eda_service import EDAService
from app.formatters.workspace_formatter import WorkspaceFormatter


def make_recommendation(
    dataset_a="customers.csv",
    dataset_b="orders.csv",
    left_column="customer_id",
    right_column="customer_id",
    cardinality="ONE_TO_MANY",
    coverage_percentage=75.0,
    recommended_join="LEFT_JOIN",
    confidence=85,
    confidence_level="HIGH",
    safe_to_execute=True,
    matched_rows=3,
    unmatched_left_rows=1,
    unmatched_right_rows=1,
    left_unique=True,
    right_unique=False,
    left_null_percentage=0.0,
    right_null_percentage=0.0,
    explanation="Test recommendation",
    evidence=None,
    warnings=None,
    all_candidates=None,
) -> JoinRecommendation:
    return JoinRecommendation(
        dataset_a=dataset_a,
        dataset_b=dataset_b,
        left_column=left_column,
        right_column=right_column,
        cardinality=cardinality,
        confidence=int(confidence),
        confidence_level=confidence_level,
        left_unique=left_unique,
        right_unique=right_unique,
        left_null_percentage=left_null_percentage,
        right_null_percentage=right_null_percentage,
        matched_rows=matched_rows,
        unmatched_left_rows=unmatched_left_rows,
        unmatched_right_rows=unmatched_right_rows,
        coverage_percentage=coverage_percentage,
        recommended_join=recommended_join,
        safe_to_execute=safe_to_execute,
        evidence=evidence or ["Valid join key"],
        warnings=warnings or [],
        explanation=explanation,
        all_candidates=all_candidates or [],
    )


class TestSafeJoinExecutor(unittest.TestCase):
    def setUp(self):
        # Canonical synthetic datasets
        # customers: C001, C002, C003
        self.customers_df = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "customer_name": ["Alice", "Bob", "Charlie"],
            "region": ["East", "West", "North"],
        })
        self.customers_df.attrs["filename"] = "customers.csv"

        # orders: O001, O002, O003, O004 (C001, C001, C002, C999)
        self.orders_df = pd.DataFrame({
            "order_id": ["O001", "O002", "O003", "O004"],
            "customer_id": ["C001", "C001", "C002", "C999"],
            "amount": [100, 200, 150, 50],
            "region": ["East", "East", "West", "South"],
        })
        self.orders_df.attrs["filename"] = "orders.csv"

        self.customers_profile = DataProfiler.generate_profile(self.customers_df, filename="customers.csv")
        self.orders_profile = DataProfiler.generate_profile(self.orders_df, filename="orders.csv")

        # Canonical recommendation from Phase 2
        self.canonical_recommendation = make_recommendation(
            dataset_a="customers.csv",
            dataset_b="orders.csv",
            left_column="customer_id",
            right_column="customer_id",
            cardinality="ONE_TO_MANY",
            coverage_percentage=75.0,
            recommended_join="LEFT_JOIN",
            confidence=85,
            confidence_level="HIGH",
            safe_to_execute=True,
            left_unique=True,
            right_unique=False,
            matched_rows=3,
            unmatched_left_rows=1,
            unmatched_right_rows=1,
        )

    # 1. successful one-to-many LEFT_JOIN
    def test_01_successful_one_to_many_left_join(self):
        rec = make_recommendation(
            dataset_a="orders.csv",
            dataset_b="customers.csv",
            left_column="customer_id",
            right_column="customer_id",
            cardinality="MANY_TO_ONE",
            coverage_percentage=75.0,
            recommended_join="LEFT_JOIN",
            confidence=85,
            confidence_level="HIGH",
            safe_to_execute=True,
            left_unique=False,
            right_unique=True,
            matched_rows=3,
            unmatched_left_rows=1,
            unmatched_right_rows=1,
        )

        output = SafeJoinExecutor.execute(self.orders_df, self.customers_df, rec)
        self.assertIn(output.result.status, ["SUCCESS", "SUCCESS_WITH_WARNINGS"])
        self.assertIsNotNone(output.dataframe)
        self.assertEqual(len(output.dataframe), 4)
        self.assertEqual(output.result.rows_before_left, 4)
        self.assertEqual(output.result.rows_before_right, 3)
        self.assertEqual(output.result.rows_after, 4)
        self.assertEqual(output.result.join_type, "LEFT_JOIN")

    # 2. successful INNER_JOIN with 100% coverage
    def test_02_successful_inner_join_100_percent_coverage(self):
        c_df = pd.DataFrame({"id": ["1", "2", "3"], "val_c": ["a", "b", "c"]})
        c_df.attrs["filename"] = "c.csv"
        o_df = pd.DataFrame({"id": ["1", "2", "3"], "val_o": [10, 20, 30]})
        o_df.attrs["filename"] = "o.csv"

        rec = make_recommendation(
            dataset_a="c.csv",
            dataset_b="o.csv",
            left_column="id",
            right_column="id",
            cardinality="ONE_TO_ONE",
            coverage_percentage=100.0,
            recommended_join="INNER_JOIN",
            confidence=95,
            confidence_level="HIGH",
            safe_to_execute=True,
            left_unique=True,
            right_unique=True,
            matched_rows=3,
            unmatched_left_rows=0,
            unmatched_right_rows=0,
        )

        output = SafeJoinExecutor.execute(c_df, o_df, rec, requested_join_type="INNER_JOIN")
        self.assertEqual(output.result.status, "SUCCESS")
        self.assertIsNotNone(output.dataframe)
        self.assertEqual(len(output.dataframe), 3)
        self.assertEqual(output.result.matched_rows, 3)
        self.assertEqual(output.result.unmatched_left_rows, 0)
        self.assertEqual(output.result.unmatched_right_rows, 0)

    # 3. successful RIGHT_JOIN where appropriate
    def test_03_successful_right_join(self):
        c_df = pd.DataFrame({"id": ["1", "2"], "val_c": ["a", "b"]})
        c_df.attrs["filename"] = "c.csv"
        o_df = pd.DataFrame({"id": ["1", "2", "3"], "val_o": [10, 20, 30]})
        o_df.attrs["filename"] = "o.csv"

        rec = make_recommendation(
            dataset_a="c.csv",
            dataset_b="o.csv",
            left_column="id",
            right_column="id",
            cardinality="ONE_TO_ONE",
            coverage_percentage=66.7,
            recommended_join="RIGHT_JOIN",
            confidence=80,
            confidence_level="HIGH",
            safe_to_execute=True,
            left_unique=True,
            right_unique=True,
            matched_rows=2,
            unmatched_left_rows=0,
            unmatched_right_rows=1,
        )

        output = SafeJoinExecutor.execute(c_df, o_df, rec, requested_join_type="RIGHT_JOIN")
        self.assertIn(output.result.status, ["SUCCESS", "SUCCESS_WITH_WARNINGS"])
        self.assertEqual(len(output.dataframe), 3)

    # 4. successful FULL_OUTER_JOIN where appropriate
    def test_04_successful_full_outer_join(self):
        c_df = pd.DataFrame({"id": ["1", "2"], "val_c": ["a", "b"]})
        c_df.attrs["filename"] = "c.csv"
        o_df = pd.DataFrame({"id": ["2", "3"], "val_o": [20, 30]})
        o_df.attrs["filename"] = "o.csv"

        rec = make_recommendation(
            dataset_a="c.csv",
            dataset_b="o.csv",
            left_column="id",
            right_column="id",
            cardinality="ONE_TO_ONE",
            coverage_percentage=50.0,
            recommended_join="FULL_OUTER_JOIN",
            confidence=80,
            confidence_level="HIGH",
            safe_to_execute=True,
            left_unique=True,
            right_unique=True,
            matched_rows=1,
            unmatched_left_rows=1,
            unmatched_right_rows=1,
        )

        output = SafeJoinExecutor.execute(c_df, o_df, rec, requested_join_type="FULL_OUTER_JOIN")
        self.assertIn(output.result.status, ["SUCCESS", "SUCCESS_WITH_WARNINGS"])
        self.assertEqual(len(output.dataframe), 3)  # id 1, 2, 3

    # 5. blocked NO_SAFE_JOIN
    def test_05_blocked_no_safe_join(self):
        rec = make_recommendation(
            dataset_a="customers.csv",
            dataset_b="orders.csv",
            left_column="customer_id",
            right_column="customer_id",
            cardinality="MANY_TO_MANY",
            coverage_percentage=20.0,
            recommended_join="NO_SAFE_JOIN",
            confidence=30,
            confidence_level="LOW",
            safe_to_execute=False,
            left_unique=False,
            right_unique=False,
            matched_rows=0,
            unmatched_left_rows=3,
            unmatched_right_rows=4,
        )

        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, rec)
        self.assertEqual(output.result.status, "BLOCKED")
        self.assertFalse(output.result.safe_to_execute)
        self.assertIsNone(output.dataframe)
        self.assertIn("NO_SAFE_JOIN", output.result.error_reason)

    # 6. blocked many-to-many
    def test_06_blocked_many_to_many(self):
        df_left = pd.DataFrame({"key": ["A", "A", "B"], "val_1": [1, 2, 3]})
        df_left.attrs["filename"] = "left.csv"
        df_right = pd.DataFrame({"key": ["A", "A", "C"], "val_2": [4, 5, 6]})
        df_right.attrs["filename"] = "right.csv"

        rec = make_recommendation(
            dataset_a="left.csv",
            dataset_b="right.csv",
            left_column="key",
            right_column="key",
            cardinality="MANY_TO_MANY",
            coverage_percentage=50.0,
            recommended_join="LEFT_JOIN",
            confidence=80,
            confidence_level="HIGH",
            safe_to_execute=True,  # Even if misconfigured as safe
            left_unique=False,
            right_unique=False,
            matched_rows=2,
            unmatched_left_rows=1,
            unmatched_right_rows=1,
        )

        output = SafeJoinExecutor.execute(df_left, df_right, rec)
        self.assertEqual(output.result.status, "BLOCKED")
        self.assertIsNone(output.dataframe)
        self.assertIn("many-to-many", output.result.error_reason.lower())

    # 7. blocked index-like key
    def test_07_blocked_index_like_key(self):
        df_left = pd.DataFrame({"index": [0, 1, 2], "val": ["x", "y", "z"]})
        df_left.attrs["filename"] = "l.csv"
        df_right = pd.DataFrame({"index": [0, 1, 2], "score": [10, 20, 30]})
        df_right.attrs["filename"] = "r.csv"

        rec = make_recommendation(
            dataset_a="l.csv",
            dataset_b="r.csv",
            left_column="index",
            right_column="index",
            cardinality="ONE_TO_ONE",
            coverage_percentage=100.0,
            recommended_join="INNER_JOIN",
            confidence=80,
            confidence_level="HIGH",
            safe_to_execute=True,
            left_unique=True,
            right_unique=True,
            matched_rows=3,
            unmatched_left_rows=0,
            unmatched_right_rows=0,
        )

        output = SafeJoinExecutor.execute(df_left, df_right, rec)
        self.assertEqual(output.result.status, "BLOCKED")
        self.assertIsNone(output.dataframe)
        self.assertIn("index-like", output.result.error_reason.lower())

    # 8. blocked incompatible key
    def test_08_blocked_incompatible_key(self):
        df_left = pd.DataFrame({"code": ["A", "B", "C"]})
        df_left.attrs["filename"] = "l.csv"
        df_right = pd.DataFrame({"num": [100.5, 200.7, 300.9]})
        df_right.attrs["filename"] = "r.csv"

        rec = make_recommendation(
            dataset_a="l.csv",
            dataset_b="r.csv",
            left_column="code",
            right_column="num",
            cardinality="ONE_TO_ONE",
            coverage_percentage=0.0,
            recommended_join="LEFT_JOIN",
            confidence=30,  # Below minimum confidence
            confidence_level="LOW",
            safe_to_execute=False,
            left_unique=True,
            right_unique=True,
            matched_rows=0,
            unmatched_left_rows=3,
            unmatched_right_rows=3,
        )

        output = SafeJoinExecutor.execute(df_left, df_right, rec)
        self.assertEqual(output.result.status, "BLOCKED")
        self.assertIsNone(output.dataframe)

    # 9. blocked invalid requested join type
    def test_09_blocked_invalid_requested_join_type(self):
        output = SafeJoinExecutor.execute(
            self.customers_df,
            self.orders_df,
            self.canonical_recommendation,
            requested_join_type="CROSS_JOIN",  # Forbidden
        )
        self.assertEqual(output.result.status, "BLOCKED")
        self.assertIsNone(output.dataframe)
        self.assertIn("Unsupported join type", output.result.error_reason)

    # 10. blocked mismatched requested key
    def test_10_blocked_mismatched_requested_key(self):
        rec = make_recommendation(
            dataset_a="customers.csv",
            dataset_b="orders.csv",
            left_column="non_existent_id",
            right_column="customer_id",
            cardinality="ONE_TO_MANY",
            coverage_percentage=75.0,
            recommended_join="LEFT_JOIN",
            confidence=85,
            confidence_level="HIGH",
            safe_to_execute=True,
            left_unique=True,
            right_unique=False,
            matched_rows=3,
            unmatched_left_rows=0,
            unmatched_right_rows=0,
        )

        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, rec)
        self.assertEqual(output.result.status, "BLOCKED")
        self.assertIsNone(output.dataframe)
        self.assertIn("does not exist", output.result.error_reason)

    # 11. input DataFrames remain unchanged
    def test_11_input_dataframes_remain_unchanged(self):
        orig_cust = self.customers_df.copy(deep=True)
        orig_orders = self.orders_df.copy(deep=True)

        _ = SafeJoinExecutor.execute(self.customers_df, self.orders_df, self.canonical_recommendation)

        pd.testing.assert_frame_equal(self.customers_df, orig_cust)
        pd.testing.assert_frame_equal(self.orders_df, orig_orders)

    # 12. correct row counts
    def test_12_correct_row_counts(self):
        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, self.canonical_recommendation)
        # customers has 3 rows (C001, C002, C003)
        # orders has 4 rows (C001, C001, C002, C999)
        # customers LEFT JOIN orders on customer_id gives 4 rows: C001(2 orders), C002(1 order), C003(0 orders -> NaN)
        self.assertEqual(output.result.rows_before_left, 3)
        self.assertEqual(output.result.rows_before_right, 4)
        self.assertEqual(output.result.rows_after, 4)

    # 13. correct unmatched counts
    def test_13_correct_unmatched_counts(self):
        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, self.canonical_recommendation)
        # C003 is not in orders -> unmatched left = 1
        # C999 is not in customers -> unmatched right = 1
        self.assertEqual(output.result.unmatched_left_rows, 1)
        self.assertEqual(output.result.unmatched_right_rows, 1)

    # 14. correct multiplication factor
    def test_14_correct_multiplication_factor(self):
        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, self.canonical_recommendation)
        # max(3, 4) = 4; rows_after = 4; factor = 4/4 = 1.0
        self.assertEqual(output.result.row_multiplication_factor, 1.0)

    # 15. duplicate expansion detection
    def test_15_duplicate_expansion_detection(self):
        # When 1:N join causes rows_after > rows_before_left
        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, self.canonical_recommendation)
        # rows_after = 4, rows_before_left = 3 -> 4 > 3 -> duplicate expansion detected
        self.assertTrue(output.result.duplicate_expansion_detected)

    # 16. zero-match warning
    def test_16_zero_match_warning(self):
        c_df = pd.DataFrame({"id": ["X1", "X2"], "val": [1, 2]})
        c_df.attrs["filename"] = "c.csv"
        o_df = pd.DataFrame({"id": ["Y1", "Y2"], "val": [3, 4]})
        o_df.attrs["filename"] = "o.csv"

        rec = make_recommendation(
            dataset_a="c.csv",
            dataset_b="o.csv",
            left_column="id",
            right_column="id",
            cardinality="ONE_TO_ONE",
            coverage_percentage=0.0,
            recommended_join="FULL_OUTER_JOIN",
            confidence=60,
            confidence_level="MODERATE",
            safe_to_execute=True,
            left_unique=True,
            right_unique=True,
            matched_rows=0,
            unmatched_left_rows=2,
            unmatched_right_rows=2,
        )

        output = SafeJoinExecutor.execute(c_df, o_df, rec)
        self.assertEqual(output.result.status, "SUCCESS_WITH_WARNINGS")
        self.assertTrue(any("zero matched" in w.lower() for w in output.result.warnings))

    # 17. overlapping column suffix handling
    def test_17_overlapping_column_suffix_handling(self):
        # Both DataFrames have 'region' in addition to 'customer_id'
        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, self.canonical_recommendation)
        self.assertIsNotNone(output.dataframe)
        cols = list(output.dataframe.columns)
        self.assertIn("region_left", cols)
        self.assertIn("region_right", cols)
        self.assertTrue(any("overlapping" in ev.lower() for ev in output.result.evidence))

    # 18. successful execution with Phase 2 recommendation
    def test_18_successful_execution_with_phase2_recommendation(self):
        # Generate real recommendation via JoinIntelligenceAnalyzer
        real_rec = JoinIntelligenceAnalyzer.analyze_pair(
            self.customers_profile,
            self.customers_df,
            self.orders_profile,
            self.orders_df,
            use_gemini=False,
        )

        self.assertTrue(real_rec.safe_to_execute)
        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, real_rec)
        self.assertIn(output.result.status, ["SUCCESS", "SUCCESS_WITH_WARNINGS"])
        self.assertIsNotNone(output.dataframe)
        self.assertEqual(len(output.dataframe), 4)

    # 19. no automatic execution during workspace analysis
    def test_19_no_automatic_execution_during_workspace_analysis(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f1, \
             tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f2:
            self.customers_df.to_csv(f1.name, index=False)
            self.orders_df.to_csv(f2.name, index=False)
            p1, p2 = f1.name, f2.name

        try:
            with patch("pandas.merge") as mock_merge:
                workspace_result = EDAService.analyze_workspace([p1, p2], use_gemini=False)
                # Ensure join intelligence recommendations are present
                self.assertIn("join_recommendations", workspace_result)
                # Ensure pandas.merge was NEVER called during workspace analysis
                mock_merge.assert_not_called()
        finally:
            if os.path.exists(p1):
                os.remove(p1)
            if os.path.exists(p2):
                os.remove(p2)

    # 20. CRITICAL NEGATIVE TEST: safe_to_execute=False means pandas.merge is NEVER called
    def test_20_critical_negative_test_merge_never_called_when_unsafe(self):
        unsafe_rec = make_recommendation(
            dataset_a="customers.csv",
            dataset_b="orders.csv",
            left_column="customer_id",
            right_column="customer_id",
            cardinality="MANY_TO_MANY",
            coverage_percentage=20.0,
            recommended_join="NO_SAFE_JOIN",
            confidence=30,
            confidence_level="LOW",
            safe_to_execute=False,  # Unsafe
            left_unique=False,
            right_unique=False,
            matched_rows=0,
            unmatched_left_rows=3,
            unmatched_right_rows=4,
        )

        with patch("pandas.merge") as mock_merge:
            output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, unsafe_rec)
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)
            self.assertFalse(output.result.safe_to_execute)

    # 21. Formatter renders execution card
    def test_21_formatter_renders_execution_card(self):
        output = SafeJoinExecutor.execute(self.customers_df, self.orders_df, self.canonical_recommendation)
        html = WorkspaceFormatter.format_join_execution(output.result)
        self.assertIn("Safe Join Executed", html)
        self.assertIn("customers.csv", html)
        self.assertIn("orders.csv", html)
        self.assertIn("customer_id", html)

    # ==============================================================
    # PHASE 3 SAFETY HARDENING: DATASET IDENTITY & PROVENANCE TESTS
    # ==============================================================

    # 22. matching dataset provenance -> execution allowed
    def test_22_matching_dataset_provenance_allowed(self):
        df_left = self.customers_df.copy()
        df_left.attrs["filename"] = "customers.csv"
        df_right = self.orders_df.copy()
        df_right.attrs["filename"] = "orders.csv"

        output = SafeJoinExecutor.execute(df_left, df_right, self.canonical_recommendation)
        self.assertIn(output.result.status, ["SUCCESS", "SUCCESS_WITH_WARNINGS"])
        self.assertIsNotNone(output.dataframe)
        self.assertEqual(len(output.dataframe), 4)

    # 23. mismatched left dataset -> BLOCKED
    def test_23_mismatched_left_dataset_blocked(self):
        unrelated_df = pd.DataFrame({"customer_id": ["C001", "C002"], "x": [1, 2]})
        unrelated_df.attrs["filename"] = "unrelated_products.csv"

        with patch("pandas.merge") as mock_merge:
            output = SafeJoinExecutor.execute(unrelated_df, self.orders_df, self.canonical_recommendation)
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)
            self.assertIn("mismatch", output.result.error_reason.lower())
            self.assertIn("unrelated_products.csv", output.result.error_reason)

    # 24. mismatched right dataset -> BLOCKED
    def test_24_mismatched_right_dataset_blocked(self):
        unrelated_df = pd.DataFrame({"customer_id": ["C001", "C002"], "y": [3, 4]})
        unrelated_df.attrs["filename"] = "unrelated_inventory.csv"

        with patch("pandas.merge") as mock_merge:
            output = SafeJoinExecutor.execute(self.customers_df, unrelated_df, self.canonical_recommendation)
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)
            self.assertIn("mismatch", output.result.error_reason.lower())
            self.assertIn("unrelated_inventory.csv", output.result.error_reason)

    # 25. unavailable/ambiguous dataset identity -> BLOCKED
    def test_25_unavailable_dataset_identity_blocked(self):
        bare_df_left = pd.DataFrame({"customer_id": ["C001", "C002"]})
        # Clear any attrs
        bare_df_left.attrs.clear()

        bare_df_right = pd.DataFrame({"customer_id": ["C001", "C002"]})
        bare_df_right.attrs.clear()

        with patch("pandas.merge") as mock_merge:
            output = SafeJoinExecutor.execute(bare_df_left, bare_df_right, self.canonical_recommendation)
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)
            self.assertIn("identity validation failed", output.result.error_reason.lower())

    # 26. critical negative test: pandas.merge is NEVER called for identity mismatch
    def test_26_critical_negative_test_merge_never_called_for_identity_mismatch(self):
        spoofed_df = pd.DataFrame({"customer_id": ["C001", "C002"], "rogue_data": [999, 888]})
        spoofed_df.attrs["filename"] = "rogue_dataset.csv"

        with patch("pandas.merge") as mock_merge:
            output = SafeJoinExecutor.execute(spoofed_df, self.orders_df, self.canonical_recommendation)
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)

    # ==============================================================
    # ANTI-SPOOFING TESTS (PROVENANCE AUTHORITATIVE OVER EXPLICIT)
    # ==============================================================

    # 27. Contradictory explicit name vs DataFrame provenance is BLOCKED
    def test_27_contradictory_explicit_name_vs_dataframe_provenance_blocked(self):
        unrelated_df = pd.DataFrame({"customer_id": ["C001", "C002"], "x": [1, 2]})
        unrelated_df.attrs["filename"] = "unrelated_products.csv"

        with patch("pandas.merge") as mock_merge:
            # Caller passes spoofed explicit left_name="customers.csv"
            output = SafeJoinExecutor.execute(
                left_df=unrelated_df,
                right_df=self.orders_df,
                recommendation=self.canonical_recommendation,
                left_name="customers.csv",
                right_name="orders.csv",
            )
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)
            self.assertIn("mismatch", output.result.error_reason.lower())
            self.assertTrue(
                "unrelated_products.csv" in output.result.error_reason
                or "contradictory" in output.result.error_reason.lower()
            )

    # 28. Matching DataFrame provenance + matching explicit name is ALLOWED
    def test_28_matching_dataframe_provenance_and_matching_explicit_name(self):
        output = SafeJoinExecutor.execute(
            left_df=self.customers_df,
            right_df=self.orders_df,
            recommendation=self.canonical_recommendation,
            left_name="customers.csv",
            right_name="orders.csv",
        )
        self.assertIn(output.result.status, ["SUCCESS", "SUCCESS_WITH_WARNINGS"])
        self.assertIsNotNone(output.dataframe)
        self.assertEqual(len(output.dataframe), 4)

    # 29. DataFrame provenance mismatch must always override explicit name
    def test_29_dataframe_provenance_mismatch_always_overrides_explicit_name(self):
        unrelated_df = pd.DataFrame({"customer_id": ["C001", "C002"], "secret": ["A", "B"]})
        unrelated_df.attrs["filename"] = "secret_payroll.csv"

        with patch("pandas.merge") as mock_merge:
            # Caller attempts to override by passing left_name="customers.csv"
            output = SafeJoinExecutor.execute(
                left_df=unrelated_df,
                right_df=self.orders_df,
                recommendation=self.canonical_recommendation,
                left_name="customers.csv",
            )
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)
            self.assertIn("mismatch", output.result.error_reason.lower())

    # 30. Contradictory explicit name on right side is BLOCKED
    def test_30_contradictory_explicit_name_right_side_blocked(self):
        unrelated_df = pd.DataFrame({"customer_id": ["C001", "C002"], "y": [10, 20]})
        unrelated_df.attrs["filename"] = "unrelated_orders.csv"

        with patch("pandas.merge") as mock_merge:
            output = SafeJoinExecutor.execute(
                left_df=self.customers_df,
                right_df=unrelated_df,
                recommendation=self.canonical_recommendation,
                left_name="customers.csv",
                right_name="orders.csv",  # Contradicts unrelated_orders.csv
            )
            mock_merge.assert_not_called()
            self.assertEqual(output.result.status, "BLOCKED")
            self.assertIsNone(output.dataframe)
            self.assertIn("mismatch", output.result.error_reason.lower())


if __name__ == "__main__":
    unittest.main()

