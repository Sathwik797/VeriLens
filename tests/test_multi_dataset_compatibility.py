import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import plotly.graph_objects as go

from app.eda.column_normalizer import ColumnNormalizer
from app.eda.dataset_relationship import DatasetCompatibilityAnalyzer
from app.eda.loader import DataLoader
from app.eda.profiler import DataProfiler
from app.models.dataset_profile import DatasetProfile
from app.models.dataset_relationship import DatasetRelationship
from app.services.eda_service import EDAService
from ui.app import analyze_dataset, process_question


class TestMultiDatasetCompatibility(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _save_csv(self, filename: str, df: pd.DataFrame) -> str:
        path = os.path.join(self.temp_dir.name, filename)
        df.to_csv(path, index=False)
        return path

    def _save_xlsx(self, filename: str, sheets_data: dict) -> str:
        path = os.path.join(self.temp_dir.name, filename)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sname, df in sheets_data.items():
                df.to_excel(writer, sheet_name=sname, index=False)
        return path

    def test_01_upload_one_csv(self):
        """1. Single CSV upload returns standard single-dataset profile card and 4 charts."""
        df = pd.DataFrame({"Category": ["A", "B"], "Sales": [100, 200]})
        path = self._save_csv("sales.csv", df)
        mock_file = MagicMock()
        mock_file.name = path

        profile_html, c1, c2, c3, c4 = analyze_dataset(file=mock_file)
        self.assertIn("Rows", profile_html)
        self.assertIn("Columns", profile_html)
        self.assertIsInstance(c1, go.Figure)

    def test_02_upload_one_xlsx(self):
        """2. Single XLSX upload returns profile card with active sheet banner and 4 charts."""
        df = pd.DataFrame({"Region": ["East", "West"], "Profit": [50, 75]})
        path = self._save_xlsx("profit.xlsx", {"Q1": df})
        mock_file = MagicMock()
        mock_file.name = path

        profile_html, c1, c2, c3, c4 = analyze_dataset(file=mock_file)
        self.assertIn("Active Sheet", profile_html)
        self.assertIn("Q1", profile_html)
        self.assertIsInstance(c1, go.Figure)

    def test_03_upload_multiple_csv_files(self):
        """3. Multiple CSV files upload produces multi-dataset workspace HTML with both files."""
        f1 = self._save_csv("cust.csv", pd.DataFrame({"customer_id": [1, 2], "name": ["A", "B"]}))
        f2 = self._save_csv("ord.csv", pd.DataFrame({"order_id": [10, 20], "customer_id": [1, 2], "amount": [99, 149]}))

        m1 = MagicMock(); m1.name = f1; m1.orig_name = "cust.csv"
        m2 = MagicMock(); m2.name = f2; m2.orig_name = "ord.csv"

        workspace_html, c1, c2, c3, c4 = analyze_dataset(files=[m1, m2])
        self.assertIn("Multi-Dataset Workspace", workspace_html)
        self.assertIn("cust.csv", workspace_html)
        self.assertIn("ord.csv", workspace_html)
        self.assertIn("Dataset Compatibility & Relationships", workspace_html)

    def test_04_upload_csv_plus_xlsx(self):
        """4. Uploading CSV + XLSX simultaneously ingests and compares both."""
        f1 = self._save_csv("items.csv", pd.DataFrame({"item_id": [101, 102], "price": [10.5, 20.0]}))
        f2 = self._save_xlsx("stock.xlsx", {"Inventory": pd.DataFrame({"item_id": [101, 102], "qty": [50, 100]})})

        m1 = MagicMock(); m1.name = f1; m1.orig_name = "items.csv"
        m2 = MagicMock(); m2.name = f2; m2.orig_name = "stock.xlsx"

        workspace_html, c1, c2, c3, c4 = analyze_dataset(files=[m1, m2])
        self.assertIn("items.csv", workspace_html)
        self.assertIn("stock.xlsx", workspace_html)
        self.assertIn("RELATED", workspace_html)

    def test_05_each_file_is_profiled_independently(self):
        """5. Each file in workspace is profiled independently without cross-contamination."""
        f1 = self._save_csv("small.csv", pd.DataFrame({"A": [1, 2]}))
        f2 = self._save_csv("large.csv", pd.DataFrame({"X": [1, 2, 3, 4, 5], "Y": ["a", "b", "c", "d", "e"]}))

        m1 = MagicMock(); m1.name = f1; m1.orig_name = "small.csv"
        m2 = MagicMock(); m2.name = f2; m2.orig_name = "large.csv"

        profiles, rels = DatasetCompatibilityAnalyzer.analyze_workspace([m1, m2], use_gemini=False)
        self.assertEqual(len(profiles), 2)
        self.assertEqual(profiles[0].rows, 2)
        self.assertEqual(profiles[0].columns, 1)
        self.assertEqual(profiles[1].rows, 5)
        self.assertEqual(profiles[1].columns, 2)

    def test_06_filename_not_used_as_relationship_evidence(self):
        """6. Deceptively similar filenames with unrelated contents are classified as UNRELATED."""
        # Filenames look like partitions, but schemas are completely unrelated
        f1 = self._save_csv("sales_part_1.csv", pd.DataFrame({"weather_temp": [72, 75], "humidity": [45, 50]}))
        f2 = self._save_csv("sales_part_2.csv", pd.DataFrame({"employee_name": ["Alice", "Bob"], "dept": ["HR", "IT"]}))

        m1 = MagicMock(); m1.name = f1; m1.orig_name = "sales_part_1.csv"
        m2 = MagicMock(); m2.name = f2; m2.orig_name = "sales_part_2.csv"

        profiles, rels = DatasetCompatibilityAnalyzer.analyze_workspace([m1, m2], use_gemini=False)
        self.assertEqual(len(rels), 1)
        self.assertEqual(rels[0].classification, "UNRELATED")
        # Evidence should not cite filename similarity
        for ev in rels[0].evidence:
            self.assertNotIn("filename", ev.lower())

    def test_07_normalized_column_names_detected(self):
        """7. Column normalizer resolves different syntaxes to canonical semantic names."""
        self.assertEqual(ColumnNormalizer.normalize("customer_id"), "customer_id")
        self.assertEqual(ColumnNormalizer.normalize("Customer ID"), "customer_id")
        self.assertEqual(ColumnNormalizer.normalize("customerId"), "customer_id")
        self.assertEqual(ColumnNormalizer.normalize("cust_id"), "customer_id")
        self.assertEqual(ColumnNormalizer.normalize("customer-id"), "customer_id")

    def test_08_same_schema_datasets_compatible(self):
        """8. Monthly partitions with identical schemas classified as COMPATIBLE."""
        df_jan = pd.DataFrame({"date": ["2023-01-01"], "product": ["Widget"], "qty": [10], "revenue": [100.0]})
        df_feb = pd.DataFrame({"date": ["2023-02-01"], "product": ["Widget"], "qty": [15], "revenue": [150.0]})

        p_jan = DataProfiler.generate_profile(df_jan, filename="sales_jan.csv")
        p_feb = DataProfiler.generate_profile(df_feb, filename="sales_feb.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_jan, df_jan, p_feb, df_feb, use_gemini=False)
        self.assertEqual(rel.classification, "COMPATIBLE")
        self.assertGreaterEqual(rel.confidence, 0.80)
        self.assertIn("append", rel.recommendation.lower())

    def test_09_related_datasets_classified_as_related(self):
        """9. Entity datasets with primary/foreign key linkage classified as RELATED."""
        df_cust = pd.DataFrame({"customer_id": [101, 102, 103], "customer_name": ["Alice", "Bob", "Charlie"], "city": ["NY", "LA", "SF"]})
        df_ord = pd.DataFrame({"order_id": [1, 2, 3, 4], "customer_id": [101, 101, 102, 103], "order_total": [50.0, 75.0, 120.0, 30.0]})

        p_cust = DataProfiler.generate_profile(df_cust, filename="customers.csv")
        p_ord = DataProfiler.generate_profile(df_ord, filename="orders.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_cust, df_cust, p_ord, df_ord, use_gemini=False)
        self.assertEqual(rel.classification, "RELATED")
        self.assertIn("customer_id", rel.candidate_keys)
        self.assertTrue(any("1-to-many" in ev for ev in rel.evidence))

    def test_10_clearly_unrelated_datasets_classified_as_unrelated(self):
        """10. Unrelated domains (weather vs employees) classified as UNRELATED."""
        df_weather = pd.DataFrame({"temperature": [65.4, 70.1], "precipitation": [0.0, 1.2], "station_code": ["S1", "S2"]})
        df_emp = pd.DataFrame({"employee_name": ["John", "Sarah"], "salary": [90000, 110000], "title": ["Engineer", "Lead"]})

        p_w = DataProfiler.generate_profile(df_weather, filename="weather.csv")
        p_e = DataProfiler.generate_profile(df_emp, filename="employees.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_w, df_weather, p_e, df_emp, use_gemini=False)
        self.assertEqual(rel.classification, "UNRELATED")
        self.assertEqual(len(rel.shared_columns), 0)

    def test_11_different_column_names_compatible_semantics(self):
        """11. Variations like cust_id and customer_id are resolved and linked."""
        df_a = pd.DataFrame({"cust_id": [1, 2, 3], "membership": ["Gold", "Silver", "Gold"]})
        df_b = pd.DataFrame({"customer_id": [1, 2, 3], "points_spent": [500, 100, 250]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rel.classification, "RELATED")
        self.assertIn("customer_id", rel.candidate_keys)

    def test_12_different_dtypes_reduces_compatibility(self):
        """12. Shared column with incompatible dtypes reduces compatibility signal."""
        df_a = pd.DataFrame({"timestamp_id": pd.date_range("2023-01-01", periods=3), "val": [10, 20, 30]})
        df_b = pd.DataFrame({"timestamp_id": ["Alpha", "Beta", "Gamma"], "val": ["x", "y", "z"]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        # Incompatible dtypes prevent COMPATIBLE classification
        self.assertNotEqual(rel.classification, "COMPATIBLE")

    def test_13_missing_columns_partial_overlap(self):
        """13. Partial overlap (e.g. 2 of 10 non-key columns) results in UNRELATED."""
        df_a = pd.DataFrame({f"col_a_{i}": [i, i+1] for i in range(10)})
        df_b = pd.DataFrame({f"col_b_{i}": [i, i+1] for i in range(10)})
        df_b["col_a_0"] = [99, 100]  # Only 1 column overlap, not a key

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rel.classification, "UNRELATED")

    def test_14_candidate_key_detection(self):
        """14. Candidate keys (unique, zero nulls) are correctly discovered."""
        df = pd.DataFrame({
            "order_id": [1001, 1002, 1003, 1004],  # Unique, key name -> candidate key
            "status": ["Shipped", "Shipped", "Pending", "Delivered"],  # Non-unique -> not key
            "amount": [12.5, 45.0, 80.0, 100.0],  # Continuous float -> not key
        })
        profile = DataProfiler.generate_profile(df)
        self.assertIn("order_id", profile.candidate_keys)
        self.assertNotIn("status", profile.candidate_keys)
        self.assertNotIn("amount", profile.candidate_keys)

    def test_15_value_overlap_signal(self):
        """15. Value overlap between datasets is computed and included in evidence."""
        df_a = pd.DataFrame({"user_id": ["u1", "u2", "u3", "u4"], "name": ["A", "B", "C", "D"]})
        df_b = pd.DataFrame({"user_id": ["u2", "u3", "u5", "u6"], "action": ["click", "login", "view", "cart"]})

        p_a = DataProfiler.generate_profile(df_a, filename="users.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="events.csv")

        rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=False)
        self.assertEqual(rel.classification, "RELATED")
        self.assertTrue(any("Value overlap" in ev for ev in rel.evidence))

    def test_16_empty_dataset_handling(self):
        """16. Empty dataset in multi-file upload is handled gracefully without exception."""
        f_valid = self._save_csv("valid.csv", pd.DataFrame({"A": [1, 2], "B": [3, 4]}))
        f_empty = self._save_csv("empty.csv", pd.DataFrame())

        m1 = MagicMock(); m1.name = f_valid; m1.orig_name = "valid.csv"
        m2 = MagicMock(); m2.name = f_empty; m2.orig_name = "empty.csv"

        profiles, rels = DatasetCompatibilityAnalyzer.analyze_workspace([m1, m2], use_gemini=False)
        # Empty dataset rejected safely by DataLoader
        self.assertEqual(len(profiles), 1)
        self.assertEqual(len(rels), 0)

    def test_17_duplicate_columns_handling(self):
        """17. Datasets with duplicate column names are normalized without crash."""
        df = pd.DataFrame([[1, 2]], columns=["col", "col"])
        profile = DataProfiler.generate_profile(df, filename="dup.csv")
        self.assertIsInstance(profile, DatasetProfile)
        self.assertEqual(profile.columns, 2)

    def test_18_gemini_unavailable_deterministic_fallback(self):
        """18. When Gemini API throws error, analyzer falls back directly to deterministic result."""
        df_a = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        df_b = pd.DataFrame({"id": [1, 2], "val": [30, 40]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.side_effect = Exception("API Quota Exceeded (429)")
            mock_model_cls.return_value = mock_model_instance

            rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=True)
            self.assertEqual(rel.classification, "COMPATIBLE")

    def test_19_gemini_invalid_json_deterministic_fallback(self):
        """19. When Gemini returns invalid JSON, analyzer falls back directly to deterministic result."""
        df_a = pd.DataFrame({"id": [1, 2], "name": ["A", "B"]})
        df_b = pd.DataFrame({"id": [1, 2], "score": [90, 80]})

        p_a = DataProfiler.generate_profile(df_a, filename="a.csv")
        p_b = DataProfiler.generate_profile(df_b, filename="b.csv")

        mock_resp = MagicMock()
        mock_resp.text = "NOT JSON ERROR RESPONSE"

        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.return_value = mock_resp
            mock_model_cls.return_value = mock_model_instance

            rel = DatasetCompatibilityAnalyzer.analyze_pair(p_a, df_a, p_b, df_b, use_gemini=True)
            self.assertEqual(rel.classification, "RELATED")

    def test_20_single_dataset_pipeline_regression(self):
        """20. Regression test: Single-dataset pipeline remains completely intact."""
        superstore_path = "datasets/superstore/Sample - Superstore.csv"
        self.assertTrue(os.path.exists(superstore_path))

        mock_file = MagicMock()
        mock_file.name = superstore_path

        profile_html, c1, c2, c3, c4 = analyze_dataset(file=mock_file)
        self.assertIn("Rows", profile_html)
        self.assertIn("9994", profile_html)
        self.assertIsInstance(c1, go.Figure)
        self.assertIsInstance(c2, go.Figure)
        self.assertIsInstance(c3, go.Figure)
        self.assertIsInstance(c4, go.Figure)


if __name__ == "__main__":
    unittest.main()
