import os
import tempfile
import unittest
from unittest.mock import MagicMock
import pandas as pd
import plotly.graph_objects as go

from app.eda.loader import DataLoader, DataLoaderError
from app.eda.profiler import DataProfiler
from app.eda.visualizer import DataVisualizer
from app.formatters.dashboard_formatter import DashboardFormatter
from app.services.eda_service import EDAService
from ui.app import analyze_dataset


class TestDataLoaderAndExcel(unittest.TestCase):

    def setUp(self):
        self.superstore_csv = "datasets/superstore/Sample - Superstore.csv"
        self.temp_dir = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.temp_dir.cleanup()

    def _create_excel(self, file_name: str, sheets_data: dict) -> str:
        """Helper to create a temporary .xlsx workbook with specified sheets and data."""
        path = os.path.join(self.temp_dir.name, file_name)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            for sheet_name, df in sheets_data.items():
                df.to_excel(writer, sheet_name=sheet_name, index=False)
        return path

    def test_01_csv_still_loads(self):
        """1. Standard CSV still loads correctly via load_csv and load_file."""
        self.assertTrue(os.path.exists(self.superstore_csv))
        df_csv = DataLoader.load_csv(self.superstore_csv)
        self.assertIsInstance(df_csv, pd.DataFrame)
        self.assertGreater(len(df_csv), 0)

        df_file = DataLoader.load_file(self.superstore_csv)
        self.assertIsInstance(df_file, pd.DataFrame)
        self.assertEqual(len(df_csv), len(df_file))

    def test_02_xlsx_loads_successfully(self):
        """2. Excel (.xlsx) loads successfully into a DataFrame with column types intact."""
        sample_df = pd.DataFrame({
            "Product": ["Laptop", "Mouse", "Keyboard", "Monitor"],
            "Revenue": [1200.0, 25.0, 75.0, 300.0],
            "Quantity": [1, 5, 3, 2],
        })
        excel_path = self._create_excel("sample.xlsx", {"SalesData": sample_df})

        df = DataLoader.load_file(excel_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 4)
        self.assertListEqual(list(df.columns), ["Product", "Revenue", "Quantity"])
        self.assertEqual(df.attrs.get("sheet_name"), "SalesData")

    def test_03_xlsx_multi_sheet_default_selection(self):
        """3. Multi-sheet workbook defaults to the first non-empty sheet."""
        empty_df = pd.DataFrame()
        valid_df1 = pd.DataFrame({"Region": ["East", "West"], "Sales": [100, 200]})
        valid_df2 = pd.DataFrame({"Region": ["North", "South"], "Sales": [300, 400]})

        excel_path = self._create_excel(
            "multi.xlsx",
            {
                "BlankSheet": empty_df,
                "FirstNonEmpty": valid_df1,
                "SecondSheet": valid_df2,
            }
        )

        df = DataLoader.load_file(excel_path)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(df.attrs.get("sheet_name"), "FirstNonEmpty")
        self.assertEqual(len(df), 2)
        self.assertEqual(list(df["Region"]), ["East", "West"])

    def test_04_xlsx_explicit_sheet_selection(self):
        """4. Specific sheet can be loaded by name via sheet_name parameter."""
        df_a = pd.DataFrame({"Category": ["Tech"], "Amount": [500]})
        df_b = pd.DataFrame({"Category": ["Office"], "Amount": [150]})

        excel_path = self._create_excel("sheets.xlsx", {"SheetA": df_a, "SheetB": df_b})

        loaded_b = DataLoader.load_file(excel_path, sheet_name="SheetB")
        self.assertEqual(loaded_b.attrs.get("sheet_name"), "SheetB")
        self.assertEqual(list(loaded_b["Category"]), ["Office"])

        # Nonexistent sheet raises DataLoaderError
        with self.assertRaises(DataLoaderError):
            DataLoader.load_file(excel_path, sheet_name="NonExistentSheet")

    def test_05_empty_excel_sheet_handled_safely(self):
        """5. Excel workbook with all empty sheets raises a clean DataLoaderError."""
        empty_df = pd.DataFrame()
        excel_path = self._create_excel("all_empty.xlsx", {"Empty1": empty_df, "Empty2": empty_df})

        with self.assertRaises(DataLoaderError) as ctx:
            DataLoader.load_file(excel_path)
        self.assertIn("no usable or non-empty sheets", str(ctx.exception).lower())

    def test_06_unsupported_extension_handled_safely(self):
        """6. Unsupported extensions (.json, .parquet) raise a clean DataLoaderError."""
        dummy_path = os.path.join(self.temp_dir.name, "data.parquet")
        with open(dummy_path, "wb") as f:
            f.write(b"PAR1dummydata")

        with self.assertRaises(DataLoaderError) as ctx:
            DataLoader.load_file(dummy_path)
        self.assertIn("Unsupported file format", str(ctx.exception))

    def test_07_invalid_or_corrupt_file_handled_safely(self):
        """7. Corrupted or invalid .xlsx file raises DataLoaderError instead of crashing."""
        corrupt_path = os.path.join(self.temp_dir.name, "broken.xlsx")
        with open(corrupt_path, "wb") as f:
            f.write(b"NOT AN EXCEL OR ZIP FILE AT ALL")

        with self.assertRaises(DataLoaderError) as ctx:
            DataLoader.load_file(corrupt_path)
        self.assertIn("Corrupted or invalid Excel file", str(ctx.exception))

    def test_08_empty_csv_handled_safely(self):
        """8. Zero-byte CSV file raises DataLoaderError."""
        empty_csv = os.path.join(self.temp_dir.name, "empty.csv")
        with open(empty_csv, "w") as f:
            pass

        with self.assertRaises(DataLoaderError) as ctx:
            DataLoader.load_file(empty_csv)
        self.assertIn("empty", str(ctx.exception).lower())

    def test_09_nonexistent_file_handled_safely(self):
        """9. Missing file path raises DataLoaderError."""
        with self.assertRaises(DataLoaderError):
            DataLoader.load_file("does/not/exist/missing.xlsx")

    def test_10_excel_dataset_profiling(self):
        """10. Excel DataFrame generates valid DatasetProfile."""
        sample_df = pd.DataFrame({
            "Department": ["Sales", "Engineering", "Marketing"],
            "Budget": [50000.0, 120000.0, 45000.0],
            "Headcount": [5, 12, 4],
        })
        excel_path = self._create_excel("corp.xlsx", {"Data": sample_df})

        df = DataLoader.load_file(excel_path)
        profile = DataProfiler.generate_profile(df)

        self.assertEqual(profile.rows, 3)
        self.assertEqual(profile.columns, 3)
        self.assertIn("Department", profile.column_names)
        self.assertIn("Budget", profile.numerical_columns)

    def test_11_excel_dataset_chart_generation(self):
        """11. Excel DataFrame passes through DataVisualizer producing valid Plotly Figures."""
        sample_df = pd.DataFrame({
            "Date": ["2023-01-15", "2023-02-15", "2023-03-15", "2023-04-15", "2023-05-15", "2023-06-15", "2023-07-15", "2023-08-15"],
            "Category": ["A", "B", "A", "B", "A", "B", "A", "B"],
            "Revenue": [100.0, 150.0, 200.0, 120.0, 180.0, 220.0, 130.0, 190.0],
        })
        excel_path = self._create_excel("trend.xlsx", {"Trends": sample_df})

        df = DataLoader.load_file(excel_path)
        charts = DataVisualizer.generate_charts(df, use_gemini=False)

        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            self.assertIn(key, charts)
            self.assertIsInstance(charts[key], go.Figure)

    def test_12_excel_dataset_through_eda_service(self):
        """12. Excel file passes end-to-end through EDAService.analyze."""
        sample_df = pd.DataFrame({
            "Region": ["North", "South", "East", "West"],
            "Sales": [1200, 850, 1400, 920],
            "Profit": [200, 110, 310, 180],
        })
        excel_path = self._create_excel("eda_test.xlsx", {"Q1_Sales": sample_df})

        result = EDAService.analyze(excel_path)
        self.assertIn("profile", result)
        self.assertEqual(result["sheet_name"], "Q1_Sales")
        self.assertEqual(result["profile"].rows, 4)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            self.assertIsInstance(result[key], go.Figure)

    def test_13_ui_analyze_dataset_with_excel(self):
        """13. UI callback analyze_dataset handles .xlsx and includes active sheet banner."""
        sample_df = pd.DataFrame({
            "Sector": ["Health", "Finance", "Tech"],
            "Index": [105.2, 110.4, 120.1],
        })
        excel_path = self._create_excel("ui_test.xlsx", {"SectorData": sample_df})

        mock_file = MagicMock()
        mock_file.name = excel_path

        profile_html, c1, c2, c3, c4 = analyze_dataset(mock_file)

        self.assertIsInstance(profile_html, str)
        self.assertIn("Active Sheet", profile_html)
        self.assertIn("SectorData", profile_html)
        self.assertIn("Rows", profile_html)
        self.assertIsInstance(c1, go.Figure)

    def test_14_ui_analyze_dataset_error_handling(self):
        """14. UI callback analyze_dataset displays friendly error card on corrupt file."""
        corrupt_path = os.path.join(self.temp_dir.name, "corrupt.xlsx")
        with open(corrupt_path, "wb") as f:
            f.write(b"not an excel file")

        mock_file = MagicMock()
        mock_file.name = corrupt_path

        profile_html, c1, c2, c3, c4 = analyze_dataset(mock_file)

        self.assertIn("Dataset Loading Error", profile_html)
        self.assertIsNone(c1)
        self.assertIsNone(c2)


if __name__ == "__main__":
    unittest.main()
