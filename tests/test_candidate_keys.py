import os
import unittest
import pandas as pd
from app.eda.column_normalizer import ColumnNormalizer
from app.eda.profiler import DataProfiler
from app.eda.loader import DataLoader
from app.eda.dataset_relationship import DatasetCompatibilityAnalyzer
from app.formatters.workspace_formatter import WorkspaceFormatter


class TestCandidateKeyDetection(unittest.TestCase):
    """
    Focused regression test suite for candidate-key detection and index-like exclusion.
    Verifies that:
    1. index is excluded
    2. Index is excluded
    3. row_number is excluded
    4. row_id is handled appropriately (excluded)
    5. Unnamed: 0 is excluded
    6. sequential numeric index is excluded
    7. customer_id remains a candidate key
    8. product_id remains a candidate key
    9. order_id remains a candidate key
    10. unrelated datasets sharing only index remain UNRELATED
    11. existing relationship detection tests still pass
    12. Real-world datasets (Amazon Sale Report, Cloud Warehouse, amazon.csv) are validated.
    """

    def test_01_index_lowercase_excluded(self):
        """1. index is excluded from candidate business keys."""
        df = pd.DataFrame({
            "index": [0, 1, 2, 3, 4],
            "customer_name": ["Alice", "Bob", "Charlie", "David", "Eve"],
            "city": ["NY", "LA", "SF", "Chicago", "Boston"],
        })
        profile = DataProfiler.generate_profile(df, filename="test.csv")
        self.assertNotIn("index", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_index_like("index", df["index"]))

    def test_02_index_titlecase_excluded(self):
        """2. Index / INDEX are excluded from candidate business keys."""
        df = pd.DataFrame({
            "Index": [1, 2, 3, 4, 5],
            "INDEX": [10, 20, 30, 40, 50],
            "metric": ["M1", "M2", "M3", "M4", "M5"],
        })
        profile = DataProfiler.generate_profile(df, filename="test.csv")
        self.assertNotIn("Index", profile.candidate_keys)
        self.assertNotIn("INDEX", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_index_like("Index", df["Index"]))
        self.assertTrue(ColumnNormalizer.is_index_like("INDEX", df["INDEX"]))

    def test_03_row_number_and_variants_excluded(self):
        """3. row_number and row_num are excluded from candidate business keys."""
        df = pd.DataFrame({
            "row_number": [1, 2, 3, 4],
            "row_num": [1, 2, 3, 4],
            "item_name": ["Pen", "Pencil", "Eraser", "Ruler"],
        })
        profile = DataProfiler.generate_profile(df, filename="test.csv")
        self.assertNotIn("row_number", profile.candidate_keys)
        self.assertNotIn("row_num", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_index_like("row_number", df["row_number"]))
        self.assertTrue(ColumnNormalizer.is_index_like("row_num", df["row_num"]))

    def test_04_row_id_handled_appropriately(self):
        """4. row_id / rowid / Row ID are excluded as technical row counters."""
        df = pd.DataFrame({
            "row_id": [1, 2, 3, 4],
            "rowid": [101, 102, 103, 104],
            "Row ID": [1, 2, 3, 4],
            "record_id": [1, 2, 3, 4],
            "record_number": [1, 2, 3, 4],
            "order_id": ["ORD-1", "ORD-2", "ORD-3", "ORD-4"],
        })
        profile = DataProfiler.generate_profile(df, filename="test.csv")
        self.assertNotIn("row_id", profile.candidate_keys)
        self.assertNotIn("rowid", profile.candidate_keys)
        self.assertNotIn("Row ID", profile.candidate_keys)
        self.assertNotIn("record_id", profile.candidate_keys)
        self.assertNotIn("record_number", profile.candidate_keys)
        self.assertIn("order_id", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_index_like("row_id", df["row_id"]))
        self.assertTrue(ColumnNormalizer.is_index_like("rowid", df["rowid"]))
        self.assertTrue(ColumnNormalizer.is_index_like("Row ID", df["Row ID"]))

    def test_05_unnamed_columns_excluded(self):
        """5. Unnamed: 0 and unnamed:0 are excluded from candidate business keys."""
        df = pd.DataFrame({
            "Unnamed: 0": [0, 1, 2, 3],
            "unnamed:0": [0, 1, 2, 3],
            "Unnamed: 22": ["x", "y", "z", "w"],
            "sku": ["SKU-001", "SKU-002", "SKU-003", "SKU-004"],
        })
        profile = DataProfiler.generate_profile(df, filename="test.csv")
        self.assertNotIn("Unnamed: 0", profile.candidate_keys)
        self.assertNotIn("unnamed:0", profile.candidate_keys)
        self.assertNotIn("Unnamed: 22", profile.candidate_keys)
        self.assertIn("sku", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_index_like("Unnamed: 0", df["Unnamed: 0"]))
        self.assertTrue(ColumnNormalizer.is_index_like("unnamed:0", df["unnamed:0"]))

    def test_06_sequential_numeric_index_excluded(self):
        """6. Obviously sequential/index-like numeric columns (e.g. 0,1,2,3,4) are excluded."""
        df = pd.DataFrame({
            "seq": [0, 1, 2, 3, 4],
            "col_counter": [1, 2, 3, 4, 5],
            "product_id": ["P-101", "P-102", "P-103", "P-104", "P-105"],
        })
        profile = DataProfiler.generate_profile(df, filename="test.csv")
        self.assertNotIn("seq", profile.candidate_keys)
        self.assertNotIn("col_counter", profile.candidate_keys)
        self.assertIn("product_id", profile.candidate_keys)

    def test_07_customer_id_remains_candidate_key(self):
        """7. customer_id and customer_number remain legitimate candidate keys."""
        df = pd.DataFrame({
            "customer_id": ["CUST-1", "CUST-2", "CUST-3"],
            "customer_number": [9001, 9002, 9003],
            "city": ["Dallas", "Austin", "Houston"],
        })
        profile = DataProfiler.generate_profile(df, filename="customers.csv")
        self.assertIn("customer_id", profile.candidate_keys)
        self.assertIn("customer_number", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_business_identifier("customer_id"))
        self.assertTrue(ColumnNormalizer.is_business_identifier("customer_number"))
        self.assertFalse(ColumnNormalizer.is_index_like("customer_id", df["customer_id"]))

    def test_08_product_id_remains_candidate_key(self):
        """8. product_id remains a candidate key even with sequential integer values."""
        df = pd.DataFrame({
            "product_id": [1, 2, 3, 4],
            "product_name": ["Laptop", "Mouse", "Keyboard", "Monitor"],
            "price": [999.0, 25.0, 75.0, 200.0],
        })
        profile = DataProfiler.generate_profile(df, filename="products.csv")
        self.assertIn("product_id", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_business_identifier("product_id"))
        self.assertFalse(ColumnNormalizer.is_index_like("product_id", df["product_id"]))

    def test_09_order_id_remains_candidate_key(self):
        """9. order_id remains a candidate key even with small sequential integer values."""
        df = pd.DataFrame({
            "order_id": [1, 2, 3, 4],
            "total_amount": [150.0, 200.0, 50.0, 320.0],
        })
        profile = DataProfiler.generate_profile(df, filename="orders.csv")
        self.assertIn("order_id", profile.candidate_keys)
        self.assertTrue(ColumnNormalizer.is_business_identifier("order_id"))
        self.assertFalse(ColumnNormalizer.is_index_like("order_id", df["order_id"]))

    def test_10_unrelated_datasets_sharing_only_index_remain_unrelated(self):
        """10. Unrelated datasets sharing only index column must NOT produce a relationship."""
        df_a = pd.DataFrame({
            "index": [0, 1, 2, 3, 4],
            "weather_condition": ["Sunny", "Rainy", "Cloudy", "Snow", "Windy"],
            "temp_f": [72, 60, 58, 28, 65],
        })
        df_b = pd.DataFrame({
            "index": [0, 1, 2, 3, 4],
            "ticker_symbol": ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"],
            "market_cap_b": [2800, 3100, 1900, 1850, 2200],
        })

        p_a = DataProfiler.generate_profile(df_a, filename="weather.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="stocks.csv")

        # index must not be in candidate keys of either dataset
        self.assertNotIn("index", p_a.candidate_keys)
        self.assertNotIn("index", p_b.candidate_keys)

        # Pairwise relationship must be UNRELATED
        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rel.classification, "UNRELATED")
        self.assertEqual(rel.candidate_keys, [])
        self.assertFalse(any("index" in k for k in rel.candidate_keys))

    def test_11_ui_workspace_formatter_never_displays_key_index(self):
        """11. UI Workspace cards must never display 'Key: index'."""
        df = pd.DataFrame({
            "index": [0, 1, 2, 3],
            "sales": [100, 200, 300, 400],
        })
        profile = DataProfiler.generate_profile(df, filename="data.csv")
        html = WorkspaceFormatter.format_workspace([profile], [])
        self.assertNotIn("Key: <code>index</code>", html)
        self.assertNotIn("🔑 Key: <code>index", html)

    def test_12_additional_business_keys_supported(self):
        """12. Additional legitimate business keys (sku, asin, invoice_id, transaction_id)."""
        df = pd.DataFrame({
            "sku": ["SKU-1", "SKU-2"],
            "asin": ["B001", "B002"],
            "invoice_id": ["INV-1", "INV-2"],
            "transaction_id": ["TXN-1", "TXN-2"],
        })
        profile = DataProfiler.generate_profile(df, filename="ecommerce.csv")
        self.assertIn("sku", profile.candidate_keys)
        self.assertIn("asin", profile.candidate_keys)
        self.assertIn("invoice_id", profile.candidate_keys)
        self.assertIn("transaction_id", profile.candidate_keys)

    def test_13_real_world_datasets_validation(self):
        """13. Validate real-world Amazon Sale Report, Cloud Warehouse, and amazon.csv."""
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

        # Amazon Sale Report must NOT report index as candidate key
        self.assertNotIn("index", prof_as.candidate_keys)
        self.assertNotIn("Unnamed: 22", prof_as.candidate_keys)

        # Cloud Warehouse must NOT report index as candidate key
        self.assertNotIn("index", prof_cw.candidate_keys)
        self.assertNotIn("Unnamed: 1", prof_cw.candidate_keys)

        # Pairwise relationships between these diverse datasets remain UNRELATED
        rel_as_cw = DatasetCompatibilityAnalyzer.analyze_pair(prof_as, df_as, prof_cw, df_cw, use_gemini=False)
        self.assertEqual(rel_as_cw.classification, "UNRELATED")
        self.assertEqual(rel_as_cw.candidate_keys, [])

        rel_as_am = DatasetCompatibilityAnalyzer.analyze_pair(prof_as, df_as, prof_am, df_am, use_gemini=False)
        self.assertEqual(rel_as_am.classification, "UNRELATED")
        self.assertEqual(rel_as_am.candidate_keys, [])


if __name__ == "__main__":
    unittest.main()
