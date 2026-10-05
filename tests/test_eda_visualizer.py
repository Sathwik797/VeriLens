import os
import unittest
import pandas as pd
import plotly.graph_objects as go

from app.eda.visualizer import DataVisualizer
from app.services.eda_service import EDAService


class TestEDAVisualizer(unittest.TestCase):

    def setUp(self):
        self.superstore_path = "datasets/superstore/Sample - Superstore.csv"
        self.walmart_path = "datasets/walmart/Walmart_Sales.csv"

    def test_01_superstore_dataset_analysis(self):
        """1. Superstore dataset generates 4 valid Plotly charts with relevant titles."""
        self.assertTrue(os.path.exists(self.superstore_path), "Superstore dataset must exist")
        result = EDAService.analyze(self.superstore_path)

        self.assertIn("profile", result)
        for key in ["sales_chart", "profit_chart", "category_chart", "region_chart"]:
            self.assertIn(key, result)
            fig = result[key]
            self.assertIsInstance(fig, go.Figure)
            self.assertTrue(len(fig.data) > 0 or len(fig.layout.annotations) > 0)
            self.assertTrue(bool(fig.layout.title.text))

        # Check that titles reflect actual columns
        self.assertIn("Sales", result["sales_chart"].layout.title.text)
        self.assertIn("Profit", result["profit_chart"].layout.title.text)
        self.assertIn("Category", result["category_chart"].layout.title.text)

    def test_02_walmart_dataset_analysis(self):
        """2. Walmart dataset (no Sales/Profit/Category/Region) generates 4 valid charts."""
        self.assertTrue(os.path.exists(self.walmart_path), "Walmart dataset must exist")
        result = EDAService.analyze(self.walmart_path)

        self.assertIn("profile", result)
        for key in ["sales_chart", "profit_chart", "category_chart", "region_chart"]:
            self.assertIn(key, result)
            fig = result[key]
            self.assertIsInstance(fig, go.Figure)
            self.assertTrue(len(fig.data) > 0)
            self.assertTrue(bool(fig.layout.title.text))

        # Check that titles reflect Walmart columns
        c1_title = result["sales_chart"].layout.title.text
        c2_title = result["profit_chart"].layout.title.text
        c3_title = result["category_chart"].layout.title.text
        c4_title = result["region_chart"].layout.title.text

        self.assertIn("Weekly_Sales", c1_title)
        self.assertIn("Distribution", c1_title)
        self.assertIn("Distribution", c2_title)
        self.assertIn("Store", c3_title)
        self.assertIn("Trend Over Time", c4_title)

    def test_03_arbitrary_unseen_column_names(self):
        """3. Dataset with completely arbitrary column names generates valid charts."""
        df = pd.DataFrame({
            "employee_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "department_name": ["Engineering", "HR", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR"],
            "annual_compensation": [120000.0, 75000.0, 95000.0, 130000.0, 80000.0, 110000.0, 140000.0, 85000.0],
            "satisfaction_rating": [4.5, 3.8, 4.2, 4.8, 3.5, 4.0, 4.9, 3.9],
            "joining_date": ["2021-01-15", "2021-03-20", "2021-06-10", "2022-01-05", "2022-04-18", "2022-08-22", "2023-02-14", "2023-05-30"],
        })

        charts = DataVisualizer.generate_charts(df)
        self.assertEqual(len(charts), 8)  # 4 generic + 4 legacy keys

        # Primary metric should be annual_compensation
        self.assertIn("annual_compensation", charts["chart_1"].layout.title.text)
        # Category should be department_name
        self.assertIn("department_name", charts["chart_3"].layout.title.text)
        # Trend should be over joining_date
        self.assertIn("Trend", charts["chart_4"].layout.title.text)

    def test_04_dataset_with_only_single_numeric_column(self):
        """4. Dataset with only one numeric column generates non-crashing charts."""
        df = pd.DataFrame({"Sensor_Reading": [12.5, 15.3, 18.2, 22.1, 19.4, 25.0, 21.8]})

        charts = DataVisualizer.generate_charts(df)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            fig = charts[key]
            self.assertIsInstance(fig, go.Figure)

    def test_05_dataset_with_only_single_categorical_column(self):
        """5. Dataset with only one categorical column generates non-crashing charts."""
        df = pd.DataFrame({"Status": ["Pending", "Approved", "Rejected", "Approved", "Pending", "Approved"]})

        charts = DataVisualizer.generate_charts(df)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            fig = charts[key]
            self.assertIsInstance(fig, go.Figure)

    def test_06_empty_and_insufficient_schema(self):
        """6. Empty DataFrames or zero-row schemas return graceful fallback figures."""
        empty_df = pd.DataFrame()
        charts_empty = DataVisualizer.generate_charts(empty_df)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            fig = charts_empty[key]
            self.assertIsInstance(fig, go.Figure)
            self.assertIn("No Data Available", fig.layout.title.text)

        zero_rows = pd.DataFrame(columns=["A", "B", "C"])
        charts_zero = DataVisualizer.generate_charts(zero_rows)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            fig = charts_zero[key]
            self.assertIsInstance(fig, go.Figure)

    def test_07_constant_and_id_only_dataset(self):
        """7. Dataset with only IDs and constant columns does not fail."""
        df = pd.DataFrame({
            "record_id": list(range(1, 20)),
            "fixed_constant": ["A"] * 19,
        })
        charts = DataVisualizer.generate_charts(df)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            fig = charts[key]
            self.assertIsInstance(fig, go.Figure)

    def test_08_backward_compatibility_legacy_methods(self):
        """8. Legacy visualizer methods still work on both Superstore and arbitrary datasets."""
        df_superstore = pd.DataFrame({
            "Sales": [100.0, 200.0, 300.0],
            "Profit": [10.0, 20.0, 30.0],
            "Category": ["A", "B", "A"],
            "Region": ["East", "West", "East"],
        })
        fig_sales = DataVisualizer.sales_distribution(df_superstore)
        fig_profit = DataVisualizer.profit_distribution(df_superstore)
        fig_cat = DataVisualizer.category_sales(df_superstore)
        fig_reg = DataVisualizer.region_profit(df_superstore)

        for fig in [fig_sales, fig_profit, fig_cat, fig_reg]:
            self.assertIsInstance(fig, go.Figure)
            self.assertGreater(len(fig.data), 0)

        # On dataset without those columns, legacy methods fall back without throwing
        df_other = pd.DataFrame({
            "revenue": [500.0, 600.0],
            "cost": [200.0, 300.0],
            "dept": ["IT", "Sales"],
        })
        self.assertIsInstance(DataVisualizer.sales_distribution(df_other), go.Figure)
        self.assertIsInstance(DataVisualizer.profit_distribution(df_other), go.Figure)
        self.assertIsInstance(DataVisualizer.category_sales(df_other), go.Figure)
        self.assertIsInstance(DataVisualizer.region_profit(df_other), go.Figure)


if __name__ == "__main__":
    unittest.main()
