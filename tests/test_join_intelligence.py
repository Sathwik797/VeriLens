import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd

from app.eda.column_normalizer import ColumnNormalizer
from app.eda.dataset_relationship import DatasetCompatibilityAnalyzer
from app.eda.join_intelligence import JoinIntelligenceAnalyzer
from app.eda.loader import DataLoader
from app.eda.profiler import DataProfiler
from app.models.join_recommendation import JoinRecommendation


class TestJoinIntelligence(unittest.TestCase):
    """
    Comprehensive test suite for Multi-Dataset Workspace Phase 2:
    Safe Join Intelligence Analyzer.
    """

    def test_01_one_to_one_relationship(self):
        """1. ONE_TO_ONE: Both sides have unique keys and 100% match."""
        df_users = pd.DataFrame({
            "user_id": ["U1", "U2", "U3"],
            "username": ["alice", "bob", "charlie"],
        })
        df_profiles = pd.DataFrame({
            "user_id": ["U1", "U2", "U3"],
            "bio": ["Engineer", "Designer", "Manager"],
        })

        p_u = DataProfiler.generate_profile(df_users, filename="users.csv")
        p_p = DataProfiler.generate_profile(df_profiles, filename="profiles.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_u, df_users, p_p, df_profiles, use_gemini=False)
        self.assertEqual(rec.cardinality, "ONE_TO_ONE")
        self.assertTrue(rec.left_unique)
        self.assertTrue(rec.right_unique)
        self.assertEqual(rec.coverage_percentage, 100.0)
        self.assertEqual(rec.recommended_join, "INNER_JOIN")
        self.assertTrue(rec.safe_to_execute)
        self.assertEqual(rec.confidence_level, "HIGH")
        self.assertGreaterEqual(rec.confidence, 90)

    def test_02_one_to_many_relationship(self):
        """2. ONE_TO_MANY: Customers (parent) to Orders (child) with 75% coverage and 1 unmatched."""
        df_cust = pd.DataFrame({
            "customer_id": ["C001", "C002", "C003"],
            "name": ["Alice", "Bob", "Charlie"],
        })
        df_ord = pd.DataFrame({
            "order_id": ["O001", "O002", "O003", "O004"],
            "customer_id": ["C001", "C001", "C002", "C999"],
            "amount": [100, 200, 150, 50],
        })

        p_cust = DataProfiler.generate_profile(df_cust, filename="customers.csv")
        p_ord = DataProfiler.generate_profile(df_ord, filename="orders.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_cust, df_cust, p_ord, df_ord, use_gemini=False)
        self.assertEqual(rec.cardinality, "ONE_TO_MANY")
        self.assertEqual(rec.left_column, "customer_id")
        self.assertEqual(rec.right_column, "customer_id")
        self.assertEqual(rec.coverage_percentage, 75.0)
        self.assertEqual(rec.matched_rows, 3)
        self.assertEqual(rec.unmatched_right_rows, 1)
        self.assertEqual(rec.recommended_join, "LEFT_JOIN")
        self.assertTrue(rec.safe_to_execute)
        self.assertEqual(rec.confidence_level, "HIGH")
        self.assertEqual(rec.confidence, 91)
        self.assertEqual(rec.parent_dataset, "customers.csv")
        self.assertEqual(rec.parent_key, "customer_id")
        self.assertEqual(rec.child_dataset, "orders.csv")
        self.assertEqual(rec.child_key, "customer_id")
        self.assertTrue(any("25.0%" in w for w in rec.warnings))

    def test_03_many_to_one_relationship(self):
        """3. MANY_TO_ONE: Orders (left, child) to Customers (right, parent)."""
        df_ord = pd.DataFrame({
            "order_id": ["O1", "O2", "O3", "O4"],
            "customer_id": ["C1", "C1", "C2", "C999"],
        })
        df_cust = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "country": ["US", "UK", "CA"],
        })

        p_ord = DataProfiler.generate_profile(df_ord, filename="orders.csv")
        p_cust = DataProfiler.generate_profile(df_cust, filename="customers.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_ord, df_ord, p_cust, df_cust, use_gemini=False)
        self.assertEqual(rec.cardinality, "MANY_TO_ONE")
        self.assertFalse(rec.left_unique)
        self.assertTrue(rec.right_unique)
        self.assertEqual(rec.coverage_percentage, 75.0)
        self.assertEqual(rec.matched_rows, 3)
        self.assertEqual(rec.unmatched_left_rows, 1)
        self.assertEqual(rec.recommended_join, "LEFT_JOIN")
        self.assertTrue(rec.safe_to_execute)
        self.assertEqual(rec.parent_dataset, "customers.csv")
        self.assertEqual(rec.child_dataset, "orders.csv")

    def test_04_many_to_many_flagged_unsafe(self):
        """4. MANY_TO_MANY: Both sides have duplicate keys -> flagged unsafe, NO_SAFE_JOIN."""
        df_a = pd.DataFrame({
            "tag_id": ["T1", "T1", "T2", "T3"],
            "article_title": ["Post 1", "Post 2", "Post 3", "Post 4"],
        })
        df_b = pd.DataFrame({
            "tag_id": ["T1", "T2", "T2", "T4"],
            "category": ["Tech", "News", "Opinion", "Science"],
        })

        p_a = DataProfiler.generate_profile(df_a, filename="articles.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="categories.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rec.cardinality, "MANY_TO_MANY")
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")
        self.assertTrue(any("Cartesian explosion" in w for w in rec.warnings))

    def test_05_incompatible_dtypes_rejected(self):
        """5. Incompatible data types (e.g. datetime vs integer) -> NO_SAFE_JOIN."""
        df_a = pd.DataFrame({
            "event_id": pd.date_range("2023-01-01", periods=3),
            "metric": [10, 20, 30],
        })
        df_b = pd.DataFrame({
            "event_id": [1, 2, 3],
            "description": ["Alpha", "Beta", "Gamma"],
        })

        p_a = DataProfiler.generate_profile(df_a, filename="events.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="descriptions.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")
        self.assertTrue(any("Incompatible data types" in w for w in rec.warnings))

    def test_06_low_coverage_rejected(self):
        """6. Low value coverage (< 50%) -> rejected, NO_SAFE_JOIN."""
        df_a = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3", "C4", "C5"],
            "city": ["NY", "LA", "SF", "CHI", "MIA"],
        })
        # Only 1 out of 5 orders matches customers (20% coverage)
        df_b = pd.DataFrame({
            "order_id": [101, 102, 103, 104, 105],
            "customer_id": ["C1", "X99", "Y88", "Z77", "W66"],
        })

        p_a = DataProfiler.generate_profile(df_a, filename="customers.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="orders.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertLess(rec.coverage_percentage, 50.0)
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")
        self.assertTrue(any("Low key value coverage" in w for w in rec.warnings))

    def test_07_high_null_rate_rejected(self):
        """7. Key column with excessive nulls (> 10%) -> rejected, NO_SAFE_JOIN."""
        df_a = pd.DataFrame({
            "customer_id": ["C1", "C2", None, None],
            "name": ["Alice", "Bob", "Charlie", "David"],
        })
        df_b = pd.DataFrame({
            "order_id": [1, 2],
            "customer_id": ["C1", "C2"],
        })

        p_a = DataProfiler.generate_profile(df_a, filename="customers.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="orders.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")
        self.assertTrue(any("Excessive null values" in w for w in rec.warnings))

    def test_08_duplicate_parent_keys_detected(self):
        """8. Purported parent table with duplicate keys -> classified as MANY_TO_MANY, rejected."""
        df_dim = pd.DataFrame({
            "customer_id": ["C1", "C1", "C2"],  # Duplicate C1
            "name": ["Alice", "Alice Duplicate", "Bob"],
        })
        df_fact = pd.DataFrame({
            "order_id": [1, 2, 3],
            "customer_id": ["C1", "C1", "C2"],
        })

        p_dim = DataProfiler.generate_profile(df_dim, filename="dim.csv")
        p_fact = DataProfiler.generate_profile(df_fact, filename="fact.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_dim, df_dim, p_fact, df_fact, use_gemini=False)
        self.assertEqual(rec.cardinality, "MANY_TO_MANY")
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")

    def test_09_index_like_key_rejected(self):
        """9. Candidate keys that are index-like (index, row_id, Unnamed: 0) are excluded."""
        df_a = pd.DataFrame({
            "index": [0, 1, 2, 3],
            "metric": [10, 20, 30, 40],
        })
        df_b = pd.DataFrame({
            "index": [0, 1, 2, 3],
            "value": [100, 200, 300, 400],
        })

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")
        self.assertNotEqual(rec.left_column, "index")

    def test_10_no_matching_key_handled(self):
        """10. Unrelated datasets with no overlapping key columns handled gracefully."""
        df_a = pd.DataFrame({"weather": ["Sunny", "Rainy"], "temperature": [75, 60]})
        df_b = pd.DataFrame({"stock_symbol": ["AAPL", "GOOG"], "price": [180, 140]})

        p_a = DataProfiler.generate_profile(df_a, filename="weather.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="stocks.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")
        self.assertEqual(rec.cardinality, "UNKNOWN")

    def test_11_multiple_candidate_keys_ranked(self):
        """11. When multiple candidate keys exist (e.g. customer_id and email), rank by confidence."""
        df_cust = pd.DataFrame({
            "customer_id": ["C1", "C2", "C3"],
            "email": ["alice@test.com", "bob@test.com", "charlie@test.com"],
            "name": ["Alice", "Bob", "Charlie"],
        })
        df_ord = pd.DataFrame({
            "order_id": [1, 2, 3],
            "customer_id": ["C1", "C2", "C3"],
            "email": ["alice@test.com", "bob@test.com", "unknown@test.com"],  # 66% on email
        })

        p_c = DataProfiler.generate_profile(df_cust, filename="cust.csv")
        p_o = DataProfiler.generate_profile(df_ord, filename="ord.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_c, df_cust, p_o, df_ord, use_gemini=False)
        self.assertGreaterEqual(len(rec.all_candidates), 2)
        # Top candidate should be customer_id due to 100% coverage
        self.assertEqual(rec.left_column, "customer_id")
        self.assertGreater(rec.all_candidates[0].confidence, rec.all_candidates[1].confidence)

    def test_12_exact_matching_key_names(self):
        """12. Exact matching key names (e.g. product_id <-> product_id) are recognized."""
        df_p = pd.DataFrame({"product_id": [101, 102], "price": [10.0, 20.0]})
        df_i = pd.DataFrame({"product_id": [101, 102], "stock": [50, 100]})

        p_p = DataProfiler.generate_profile(df_p, filename="products.csv")
        p_i = DataProfiler.generate_profile(df_i, filename="inventory.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_p, df_p, p_i, df_i, use_gemini=False)
        self.assertEqual(rec.left_column, "product_id")
        self.assertEqual(rec.right_column, "product_id")
        self.assertTrue(rec.safe_to_execute)

    def test_13_normalized_key_names(self):
        """13. Normalized key names (cust_id <-> customer_id) are resolved and matched."""
        df_a = pd.DataFrame({"cust_id": ["C1", "C2", "C3"], "segment": ["Consumer", "Corporate", "Home"]})
        df_b = pd.DataFrame({"customer_id": ["C1", "C2", "C2"], "sales": [100, 50, 75]})

        p_a = DataProfiler.generate_profile(df_a, filename="cust.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="orders.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rec.left_column, "cust_id")
        self.assertEqual(rec.right_column, "customer_id")
        self.assertEqual(rec.cardinality, "ONE_TO_MANY")
        self.assertTrue(rec.safe_to_execute)

    def test_14_candidate_key_with_partial_coverage(self):
        """14. Partial coverage metrics (matched rows, unmatched rows, percentages) are accurate."""
        df_a = pd.DataFrame({"user_id": ["U1", "U2", "U3", "U4"]})
        df_b = pd.DataFrame({"user_id": ["U1", "U2", "U5", "U6", "U7"]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rec.matched_rows, 2)
        self.assertEqual(rec.unmatched_left_rows, 2)
        self.assertEqual(rec.unmatched_right_rows, 3)

    def test_15_unrelated_datasets_rejected(self):
        """15. Datasets pre-classified as UNRELATED return safe rejection."""
        df_w = pd.DataFrame({"temp": [70, 75], "city": ["NY", "LA"]})
        df_s = pd.DataFrame({"ticker": ["AAPL", "GOOG"], "price": [150, 100]})

        p_w = DataProfiler.generate_profile(df_w, filename="weather.csv")
        p_s = DataProfiler.generate_profile(df_s, filename="stocks.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_w, df_w, p_s, df_s, use_gemini=False)
        self.assertEqual(rel.classification, "UNRELATED")

        rec = JoinIntelligenceAnalyzer.analyze_pair(p_w, df_w, p_s, df_s, relationship=rel, use_gemini=False)
        self.assertFalse(rec.safe_to_execute)
        self.assertEqual(rec.recommended_join, "NO_SAFE_JOIN")
        self.assertEqual(rec.confidence, 0)

    def test_16_gemini_unavailable_fallback(self):
        """16. When Gemini API key is missing or call fails, deterministic explanation is used."""
        df_a = pd.DataFrame({"id": ["A1", "A2"], "name": ["Alpha", "Beta"]})
        df_b = pd.DataFrame({"id": ["A1", "A2"], "score": [90, 80]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        with patch.dict(os.environ, {"GEMINI_API_KEY": ""}, clear=True):
            rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=True)
            self.assertTrue(rec.safe_to_execute)
            self.assertIn("100.0% key coverage", rec.explanation)

    def test_17_gemini_invalid_json_fallback(self):
        """17. When Gemini returns invalid JSON, deterministic explanation is preserved."""
        df_a = pd.DataFrame({"id": ["A1", "A2"], "name": ["Alpha", "Beta"]})
        df_b = pd.DataFrame({"id": ["A1", "A2"], "score": [90, 80]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        mock_resp = MagicMock()
        mock_resp.text = "NOT JSON ERROR"

        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_inst = MagicMock()
            mock_inst.generate_content.return_value = mock_resp
            mock_model_cls.return_value = mock_inst

            rec = JoinIntelligenceAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=True)
            self.assertTrue(rec.safe_to_execute)
            self.assertIn("key coverage", rec.explanation)

    def test_18_existing_phase1_regression(self):
        """18. Phase 1 compatibility classification remains intact."""
        df_a = pd.DataFrame({"col_1": [1, 2], "col_2": ["a", "b"]})
        df_b = pd.DataFrame({"col_1": [3, 4], "col_2": ["c", "d"]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rel.classification, "COMPATIBLE")

    def test_19_real_world_datasets_unrelated(self):
        """19. Validate that real-world datasets remain UNRELATED and receive NO_SAFE_JOIN."""
        real_datasets_dir = r"C:\Users\HP 440 G8\Downloads\datasets"
        as_path = os.path.join(real_datasets_dir, "Amazon Sale Report.csv")
        cw_path = os.path.join(real_datasets_dir, "Cloud Warehouse Compersion Chart.csv")
        am_path = os.path.join(real_datasets_dir, "amazon.csv")

        if not (os.path.exists(as_path) and os.path.exists(cw_path) and os.path.exists(am_path)):
            self.skipTest("Real-world datasets not found in local Downloads folder.")

        df_as = DataLoader.load_file(as_path)
        df_cw = DataLoader.load_file(cw_path)
        df_am = DataLoader.load_file(am_path)

        prof_as = DataProfiler.generate_profile(df_as, filename="Amazon Sale Report.csv")
        prof_cw = DataProfiler.generate_profile(df_cw, filename="Cloud Warehouse Compersion Chart.csv")
        prof_am = DataProfiler.generate_profile(df_am, filename="amazon.csv")

        # Classifications between Amazon Sale Report and Cloud Warehouse
        rel_as_cw = DatasetCompatibilityAnalyzer.analyze_pair(prof_as, df_as, prof_cw, df_cw, use_gemini=False)
        self.assertEqual(rel_as_cw.classification, "UNRELATED")

        rec_as_cw = JoinIntelligenceAnalyzer.analyze_pair(prof_as, df_as, prof_cw, df_cw, relationship=rel_as_cw, use_gemini=False)
        self.assertFalse(rec_as_cw.safe_to_execute)
        self.assertEqual(rec_as_cw.recommended_join, "NO_SAFE_JOIN")

        # Classifications between Amazon Sale Report and amazon.csv
        rel_as_am = DatasetCompatibilityAnalyzer.analyze_pair(prof_as, df_as, prof_am, df_am, use_gemini=False)
        self.assertEqual(rel_as_am.classification, "UNRELATED")

        rec_as_am = JoinIntelligenceAnalyzer.analyze_pair(prof_as, df_as, prof_am, df_am, relationship=rel_as_am, use_gemini=False)
        self.assertFalse(rec_as_am.safe_to_execute)
        self.assertEqual(rec_as_am.recommended_join, "NO_SAFE_JOIN")


if __name__ == "__main__":
    unittest.main()
