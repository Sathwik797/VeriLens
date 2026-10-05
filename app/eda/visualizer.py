import os
from typing import Dict, List, Optional
import pandas as pd
import plotly.graph_objects as go

from app.eda.intelligence import (
    CandidateGenerator,
    ChartRenderer,
    VisualizationCandidate,
    VisualizationRanker,
)


class DataVisualizer:
    """
    Visualization Intelligence Engine for VeriLens AI.
    Combines deterministic schema/candidate generation, Gemini semantic ranking,
    pure Pandas deterministic computation, and Plotly rendering to generate
    a visually diverse, insightful set of visualizations for any dataset.
    """

    # ---------------------------------------------------------
    # Column Candidate Proxies (Preserves Backward Compatibility)
    # ---------------------------------------------------------

    @classmethod
    def get_numeric_candidates(cls, df: pd.DataFrame) -> List[str]:
        return CandidateGenerator.get_numeric_columns(df)

    @classmethod
    def get_categorical_candidates(cls, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
        cat_cols = CandidateGenerator.get_categorical_columns(df)
        if exclude_cols:
            exclude_set = set(exclude_cols)
            return [c for c in cat_cols if c not in exclude_set]
        return cat_cols

    @classmethod
    def get_datetime_candidates(cls, df: pd.DataFrame) -> List[str]:
        return CandidateGenerator.get_datetime_columns(df)

    # ---------------------------------------------------------
    # Main Intelligence Pipeline: Generate Diverse Charts
    # ---------------------------------------------------------

    @classmethod
    def generate_charts(cls, df: pd.DataFrame, use_gemini: bool = True) -> Dict[str, go.Figure]:
        """
        Executes the visualization intelligence pipeline:
        1. Generates valid deterministic visualization candidates (bar, donut, line, scatter, histogram).
        2. Ranks/selects candidates using Gemini (with deterministic diversity fallback).
        3. Computes aggregations with Pandas.
        4. Renders interactive Plotly figures.
        """
        if df is None or df.empty:
            empty = ChartRenderer.create_empty_chart("No Data Available")
            return {
                "chart_1": empty,
                "chart_2": empty,
                "chart_3": empty,
                "chart_4": empty,
                "sales_chart": empty,
                "profit_chart": empty,
                "category_chart": empty,
                "region_chart": empty,
            }

        # 1. Generate candidates
        candidates = CandidateGenerator.generate_candidates(df)

        # 2. Select diverse candidates via Gemini or deterministic fallback
        selected_candidates: List[VisualizationCandidate] = []
        if candidates:
            if use_gemini and os.getenv("GEMINI_API_KEY"):
                col_names = df.columns.tolist()
                dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
                selected_candidates = VisualizationRanker.rank_with_gemini(
                    candidates=candidates,
                    column_names=col_names,
                    data_types=dtypes,
                    limit=4,
                )
            else:
                selected_candidates = VisualizationRanker.select_diverse_candidates(candidates, limit=4)

        # 3. Render figures for selected candidates
        figures: List[go.Figure] = []
        for cand in selected_candidates:
            figures.append(ChartRenderer.render(df, cand))

        # 4. If fewer than 4 candidates generated, fill remaining slots gracefully
        while len(figures) < 4:
            figures.append(ChartRenderer.create_empty_chart(f"Chart {len(figures) + 1} (No suitable columns)"))

        c1, c2, c3, c4 = figures[0], figures[1], figures[2], figures[3]

        return {
            "chart_1": c1,
            "chart_2": c2,
            "chart_3": c3,
            "chart_4": c4,
            "sales_chart": c1,
            "profit_chart": c2,
            "category_chart": c3,
            "region_chart": c4,
        }

    # ---------------------------------------------------------
    # Backward Compatibility Methods for Existing Callers / Tests
    # ---------------------------------------------------------

    @classmethod
    def create_distribution_chart(cls, df: pd.DataFrame, col: str) -> go.Figure:
        return ChartRenderer._render_histogram(df, col, f"{col} Distribution")

    @classmethod
    def create_category_bar_chart(cls, df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
        return ChartRenderer._render_bar(df, cat_col, num_col, "sum", f"{num_col} by {cat_col}")

    @classmethod
    def create_time_trend_chart(cls, df: pd.DataFrame, date_col: str, num_col: str) -> go.Figure:
        return ChartRenderer._render_line(df, date_col, num_col, f"{num_col} Trend Over Time")

    @classmethod
    def create_category_frequency_chart(cls, df: pd.DataFrame, cat_col: str) -> go.Figure:
        return ChartRenderer._render_bar(df, cat_col, cat_col, "count", f"{cat_col} Frequency")

    @classmethod
    def create_empty_chart(cls, title: str = "Insufficient Data") -> go.Figure:
        return ChartRenderer.create_empty_chart(title)

    @classmethod
    def sales_distribution(cls, df: pd.DataFrame) -> go.Figure:
        if "Sales" in df.columns:
            return cls.create_distribution_chart(df, "Sales")
        num_candidates = cls.get_numeric_candidates(df)
        if num_candidates:
            return cls.create_distribution_chart(df, num_candidates[0])
        return cls.create_empty_chart("Sales Distribution (Column not found)")

    @classmethod
    def profit_distribution(cls, df: pd.DataFrame) -> go.Figure:
        if "Profit" in df.columns:
            return cls.create_distribution_chart(df, "Profit")
        num_candidates = cls.get_numeric_candidates(df)
        col = num_candidates[1] if len(num_candidates) > 1 else (num_candidates[0] if num_candidates else None)
        if col:
            return cls.create_distribution_chart(df, col)
        return cls.create_empty_chart("Profit Distribution (Column not found)")

    @classmethod
    def category_sales(cls, df: pd.DataFrame) -> go.Figure:
        if "Category" in df.columns and "Sales" in df.columns:
            return cls.create_category_bar_chart(df, "Category", "Sales")
        cat_candidates = cls.get_categorical_candidates(df)
        num_candidates = cls.get_numeric_candidates(df)
        if cat_candidates and num_candidates:
            return cls.create_category_bar_chart(df, cat_candidates[0], num_candidates[0])
        return cls.create_empty_chart("Category Sales (Columns not found)")

    @classmethod
    def region_profit(cls, df: pd.DataFrame) -> go.Figure:
        if "Region" in df.columns and "Profit" in df.columns:
            return cls.create_category_bar_chart(df, "Region", "Profit")
        cat_candidates = cls.get_categorical_candidates(df)
        num_candidates = cls.get_numeric_candidates(df)
        cat = cat_candidates[1] if len(cat_candidates) > 1 else (cat_candidates[0] if cat_candidates else None)
        num = num_candidates[1] if len(num_candidates) > 1 else (num_candidates[0] if num_candidates else None)
        if cat and num:
            return cls.create_category_bar_chart(df, cat, num)
        return cls.create_empty_chart("Region Profit (Columns not found)")