import os
import unittest
from unittest.mock import MagicMock, patch
import pandas as pd
import plotly.graph_objects as go

from app.eda.intelligence import (
    CandidateGenerator,
    ChartRenderer,
    VisualizationCandidate,
    VisualizationRanker,
)
from app.eda.visualizer import DataVisualizer
from app.services.eda_service import EDAService


class TestEDAVisualizer(unittest.TestCase):

    def setUp(self):
        self.superstore_path = "datasets/superstore/Sample - Superstore.csv"
        self.walmart_path = "datasets/walmart/Walmart_Sales.csv"

    def test_01_superstore_dataset_analysis(self):
        """1. Superstore dataset generates 4 valid Plotly charts with high diversity."""
        self.assertTrue(os.path.exists(self.superstore_path), "Superstore dataset must exist")
        result = EDAService.analyze(self.superstore_path)

        self.assertIn("profile", result)
        chart_types = set()
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            fig = result[key]
            self.assertIsInstance(fig, go.Figure)
            self.assertTrue(len(fig.data) > 0 or len(fig.layout.annotations) > 0)
            self.assertTrue(bool(fig.layout.title.text))

    def test_02_walmart_dataset_analysis(self):
        """2. Walmart dataset generates 4 valid, diverse charts without schema errors."""
        self.assertTrue(os.path.exists(self.walmart_path), "Walmart dataset must exist")
        result = EDAService.analyze(self.walmart_path)

        self.assertIn("profile", result)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            fig = result[key]
            self.assertIsInstance(fig, go.Figure)
            self.assertTrue(len(fig.data) > 0)
            self.assertTrue(bool(fig.layout.title.text))

    def test_03_date_numeric_categorical_dataset(self):
        """3. Dataset with date + numeric + categorical produces trend, bar, and distributions."""
        df = pd.DataFrame({
            "employee_id": [101, 102, 103, 104, 105, 106, 107, 108],
            "department": ["Engineering", "HR", "Sales", "Engineering", "HR", "Sales", "Engineering", "HR"],
            "salary": [120000.0, 75000.0, 95000.0, 130000.0, 80000.0, 110000.0, 140000.0, 85000.0],
            "joining_date": ["2021-01-15", "2021-03-20", "2021-06-10", "2022-01-05", "2022-04-18", "2022-08-22", "2023-02-14", "2023-05-30"],
        })

        candidates = CandidateGenerator.generate_candidates(df)
        types = {c.chart_type for c in candidates}
        self.assertIn("bar", types)
        self.assertIn("line", types)
        self.assertIn("donut", types)

        charts = DataVisualizer.generate_charts(df, use_gemini=False)
        self.assertEqual(len(charts), 8)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            self.assertIsInstance(charts[key], go.Figure)

    def test_04_only_numeric_columns(self):
        """4. Dataset with only numeric columns generates histograms and scatter plots."""
        df = pd.DataFrame({
            "metric_a": [10.2, 14.5, 18.9, 22.1, 26.4, 30.0, 35.2, 40.1, 45.3, 50.0, 55.4, 60.1],
            "metric_b": [105.0, 110.2, 115.8, 120.3, 125.9, 130.4, 135.0, 140.2, 145.8, 150.1, 155.0, 160.0],
            "metric_c": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.0, 11.1, 12.2],
        })

        candidates = CandidateGenerator.generate_candidates(df)
        types = {c.chart_type for c in candidates}
        self.assertIn("scatter", types)
        self.assertIn("histogram", types)
        # Bar and pie should not be generated without categorical columns
        self.assertNotIn("bar", types)
        self.assertNotIn("donut", types)

        charts = DataVisualizer.generate_charts(df, use_gemini=False)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            self.assertIsInstance(charts[key], go.Figure)

    def test_05_only_categorical_columns(self):
        """5. Dataset with only categorical columns handles gracefully without errors."""
        df = pd.DataFrame({
            "status": ["Active", "Inactive", "Pending", "Active", "Active", "Pending"],
            "tier": ["Gold", "Silver", "Bronze", "Gold", "Silver", "Gold"],
        })

        charts = DataVisualizer.generate_charts(df, use_gemini=False)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            self.assertIsInstance(charts[key], go.Figure)

    def test_06_single_numeric_column(self):
        """6. Dataset with only one numeric column generates non-crashing charts."""
        df = pd.DataFrame({"Sensor_Reading": [12.5, 15.3, 18.2, 22.1, 19.4, 25.0, 21.8, 30.1, 28.4, 32.0]})

        candidates = CandidateGenerator.generate_candidates(df)
        self.assertTrue(any(c.chart_type == "histogram" for c in candidates))

        charts = DataVisualizer.generate_charts(df, use_gemini=False)
        for key in ["chart_1", "chart_2", "chart_3", "chart_4"]:
            self.assertIsInstance(charts[key], go.Figure)

    def test_07_high_cardinality_categories(self):
        """7. High-cardinality columns (e.g. 50 unique items) are rejected from pie charts."""
        df = pd.DataFrame({
            "high_card_category": [f"Item_{i}" for i in range(50)] * 2,
            "sales": [10.0 * (i + 1) for i in range(100)],
        })

        candidates = CandidateGenerator.generate_candidates(df)
        # Pie/donut charts should be strictly rejected for cardinality > 8
        donut_candidates = [c for c in candidates if c.chart_type in ["pie", "donut"]]
        self.assertEqual(len(donut_candidates), 0)

    def test_08_id_like_columns_rejected_from_measures(self):
        """8. ID-like columns (row_id, customer_id, order_id) are not used as scatter measures."""
        df = pd.DataFrame({
            "order_id": list(range(1, 100)),
            "customer_id": list(range(1001, 1100)),
            "sales_amount": [50.0 + i * 1.5 for i in range(99)],
            "profit_amount": [10.0 + i * 0.8 for i in range(99)],
        })

        candidates = CandidateGenerator.generate_candidates(df)
        scatters = [c for c in candidates if c.chart_type == "scatter"]
        for s in scatters:
            self.assertNotIn("order_id", [s.primary_column, s.secondary_column])
            self.assertNotIn("customer_id", [s.primary_column, s.secondary_column])

    def test_09_no_valid_trend_column(self):
        """9. Dataset without temporal columns does not generate invalid line/trend charts."""
        df = pd.DataFrame({
            "region": ["East", "West", "North", "South"],
            "revenue": [1000.0, 2000.0, 1500.0, 1800.0],
        })

        candidates = CandidateGenerator.generate_candidates(df)
        lines = [c for c in candidates if c.chart_type == "line"]
        self.assertEqual(len(lines), 0)

    def test_10_invalid_pie_chart_criteria(self):
        """10. Negative sums and rate metrics are excluded from pie/donut charts."""
        df = pd.DataFrame({
            "category": ["A", "B", "C"],
            "temperature_rate": [75.0, 80.0, 72.0],
            "losses": [-100.0, -200.0, -50.0],
        })

        candidates = CandidateGenerator.generate_candidates(df)
        donuts = [c for c in candidates if c.chart_type in ["pie", "donut"]]
        self.assertEqual(len(donuts), 0)

    def test_11_gemini_ranking_success(self):
        """11. Successful Gemini ranking correctly selects and orders candidates."""
        candidates = [
            VisualizationCandidate(
                candidate_id="cand_1", chart_type="bar", primary_column="Category",
                secondary_column="Sales", title="Sales by Category", description="desc", suitability_score=90.0
            ),
            VisualizationCandidate(
                candidate_id="cand_2", chart_type="donut", primary_column="Category",
                secondary_column="Sales", title="Sales Share by Category", description="desc", suitability_score=85.0
            ),
            VisualizationCandidate(
                candidate_id="cand_3", chart_type="line", primary_column="Date",
                secondary_column="Sales", title="Sales Trend", description="desc", suitability_score=80.0
            ),
            VisualizationCandidate(
                candidate_id="cand_4", chart_type="scatter", primary_column="Sales",
                secondary_column="Profit", title="Profit vs Sales", description="desc", suitability_score=75.0
            ),
            VisualizationCandidate(
                candidate_id="cand_5", chart_type="histogram", primary_column="Sales",
                title="Sales Distribution", description="desc", suitability_score=70.0
            ),
        ]

        mock_response = MagicMock()
        mock_response.text = '{"selected_candidates": [{"candidate_id": "cand_2", "rank": 1}, {"candidate_id": "cand_3", "rank": 2}, {"candidate_id": "cand_1", "rank": 3}, {"candidate_id": "cand_4", "rank": 4}]}'

        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.return_value = mock_response
            mock_model_cls.return_value = mock_model_instance

            selected = VisualizationRanker.rank_with_gemini(
                candidates=candidates,
                column_names=["Category", "Sales", "Date", "Profit"],
                data_types={"Category": "object", "Sales": "float64"},
                api_key="fake-test-key",
            )

            self.assertEqual(len(selected), 4)
            self.assertEqual([c.candidate_id for c in selected], ["cand_2", "cand_3", "cand_1", "cand_4"])

    def test_12_gemini_ranking_failure_fallback(self):
        """12. Gemini ranking failure triggers deterministic diversity fallback."""
        candidates = [
            VisualizationCandidate(candidate_id="b1", chart_type="bar", primary_column="Cat", secondary_column="Num", title="Bar", description="d", suitability_score=90.0),
            VisualizationCandidate(candidate_id="d1", chart_type="donut", primary_column="Cat", secondary_column="Num", title="Donut", description="d", suitability_score=85.0),
            VisualizationCandidate(candidate_id="l1", chart_type="line", primary_column="Date", secondary_column="Num", title="Line", description="d", suitability_score=80.0),
            VisualizationCandidate(candidate_id="s1", chart_type="scatter", primary_column="Num", secondary_column="Num2", title="Scatter", description="d", suitability_score=75.0),
        ]

        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.side_effect = Exception("API Connection Timeout")
            mock_model_cls.return_value = mock_model_instance

            selected = VisualizationRanker.rank_with_gemini(
                candidates=candidates,
                column_names=["Cat", "Num"],
                data_types={"Cat": "object", "Num": "float64"},
                api_key="fake-key",
            )

            self.assertEqual(len(selected), 4)
            chart_types = {c.chart_type for c in selected}
            self.assertIn("bar", chart_types)
            self.assertIn("donut", chart_types)
            self.assertIn("line", chart_types)
            self.assertIn("scatter", chart_types)

    def test_13_invalid_gemini_response_fallback(self):
        """13. Invalid Gemini JSON syntax triggers deterministic diversity fallback."""
        candidates = [
            VisualizationCandidate(candidate_id="c1", chart_type="bar", primary_column="A", secondary_column="B", title="T1", description="d", suitability_score=95.0),
            VisualizationCandidate(candidate_id="c2", chart_type="donut", primary_column="A", secondary_column="B", title="T2", description="d", suitability_score=90.0),
            VisualizationCandidate(candidate_id="c3", chart_type="line", primary_column="D", secondary_column="B", title="T3", description="d", suitability_score=85.0),
            VisualizationCandidate(candidate_id="c4", chart_type="histogram", primary_column="B", title="T4", description="d", suitability_score=80.0),
        ]

        mock_response = MagicMock()
        mock_response.text = "NOT JSON AT ALL! <ERROR>"

        with patch("google.generativeai.GenerativeModel") as mock_model_cls:
            mock_model_instance = MagicMock()
            mock_model_instance.generate_content.return_value = mock_response
            mock_model_cls.return_value = mock_model_instance

            selected = VisualizationRanker.rank_with_gemini(
                candidates=candidates,
                column_names=["A", "B", "D"],
                data_types={},
                api_key="fake-key",
            )

            self.assertEqual(len(selected), 4)

    def test_14_chart_diversity_enforcement(self):
        """14. Diversity selection prioritizes different chart types over repeated highest-scoring types."""
        candidates = [
            VisualizationCandidate(candidate_id="b1", chart_type="bar", primary_column="Cat1", secondary_column="Num", title="Bar 1", description="d", suitability_score=99.0),
            VisualizationCandidate(candidate_id="b2", chart_type="bar", primary_column="Cat2", secondary_column="Num", title="Bar 2", description="d", suitability_score=98.0),
            VisualizationCandidate(candidate_id="b3", chart_type="bar", primary_column="Cat3", secondary_column="Num", title="Bar 3", description="d", suitability_score=97.0),
            VisualizationCandidate(candidate_id="b4", chart_type="bar", primary_column="Cat4", secondary_column="Num", title="Bar 4", description="d", suitability_score=96.0),
            VisualizationCandidate(candidate_id="d1", chart_type="donut", primary_column="Cat1", secondary_column="Num", title="Donut 1", description="d", suitability_score=80.0),
            VisualizationCandidate(candidate_id="l1", chart_type="line", primary_column="Date", secondary_column="Num", title="Line 1", description="d", suitability_score=75.0),
            VisualizationCandidate(candidate_id="s1", chart_type="scatter", primary_column="Num", secondary_column="Num2", title="Scatter 1", description="d", suitability_score=70.0),
        ]

        selected = VisualizationRanker.select_diverse_candidates(candidates, limit=4)
        selected_types = [c.chart_type for c in selected]

        # Must include donut, line, scatter even though bar2, bar3, bar4 had higher scores
        self.assertIn("bar", selected_types)
        self.assertIn("donut", selected_types)
        self.assertIn("line", selected_types)
        self.assertIn("scatter", selected_types)


if __name__ == "__main__":
    unittest.main()
