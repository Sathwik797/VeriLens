from typing import Any, Dict, List
import logging
import os
import pandas as pd
from app.eda.dataset_relationship import DatasetCompatibilityAnalyzer
from app.eda.join_intelligence import JoinIntelligenceAnalyzer
from app.eda.loader import DataLoader
from app.eda.profiler import DataProfiler
from app.eda.visualizer import DataVisualizer

logger = logging.getLogger(__name__)


class EDAService:

    @staticmethod
    def analyze(file_path: str):
        """Single-dataset EDA analysis pipeline."""
        df = DataLoader.load_file(file_path)

        profile = DataProfiler.generate_profile(df, filename=file_path)

        charts = DataVisualizer.generate_charts(df)

        return {
            "profile": profile,
            "sheet_name": df.attrs.get("sheet_name"),
            "sales_chart": charts["sales_chart"],
            "profit_chart": charts["profit_chart"],
            "category_chart": charts["category_chart"],
            "region_chart": charts["region_chart"],
            "chart_1": charts["chart_1"],
            "chart_2": charts["chart_2"],
            "chart_3": charts["chart_3"],
            "chart_4": charts["chart_4"],
        }

    @classmethod
    def analyze_workspace(cls, file_paths: List[Any], use_gemini: bool = True) -> Dict[str, Any]:
        """
        Multi-dataset workspace analysis:
        1. Profiles each uploaded dataset independently.
        2. Determines pairwise compatibility and relationships.
        3. For RELATED pairs, runs JoinIntelligenceAnalyzer for safe join recommendations.
        4. Renders diverse EDA visualizations for the primary dataset.
        """
        profiles, relationships = DatasetCompatibilityAnalyzer.analyze_workspace(
            file_paths, use_gemini=use_gemini
        )

        # Cache dataframes for primary chart generation and join intelligence
        dfs: Dict[str, pd.DataFrame] = {}
        for f in file_paths:
            path = f.name if hasattr(f, "name") else str(f)
            orig_name = getattr(f, "orig_name", None) or os.path.basename(path)
            try:
                dfs[orig_name] = DataLoader.load_file(path)
            except Exception:
                pass

        # Compute safe join recommendations for RELATED dataset pairs
        join_recommendations = []
        for rel in relationships:
            if rel.classification == "RELATED":
                df_a = dfs.get(rel.dataset_a)
                df_b = dfs.get(rel.dataset_b)
                p_a = next((p for p in profiles if p.filename == rel.dataset_a), None)
                p_b = next((p for p in profiles if p.filename == rel.dataset_b), None)
                if df_a is not None and df_b is not None and p_a is not None and p_b is not None:
                    try:
                        join_rec = JoinIntelligenceAnalyzer.analyze_pair(
                            p_a, df_a, p_b, df_b, relationship=rel, use_gemini=use_gemini
                        )
                        join_recommendations.append(join_rec)
                    except Exception as e:
                        logger.error(f"Join intelligence failed for {rel.dataset_a} <-> {rel.dataset_b}: {e}")

        primary_charts: Dict[str, Any] = {}
        if file_paths:
            primary_file = file_paths[0]
            primary_path = primary_file.name if hasattr(primary_file, "name") else str(primary_file)
            primary_name = getattr(primary_file, "orig_name", None) or os.path.basename(primary_path)
            df_primary = dfs.get(primary_name)
            if df_primary is not None:
                try:
                    primary_charts = DataVisualizer.generate_charts(df_primary)
                except Exception:
                    pass

        empty_fig = DataVisualizer.create_empty_chart("No chart")
        return {
            "profiles": profiles,
            "relationships": relationships,
            "join_recommendations": join_recommendations,
            "chart_1": primary_charts.get("chart_1", empty_fig),
            "chart_2": primary_charts.get("chart_2", empty_fig),
            "chart_3": primary_charts.get("chart_3", empty_fig),
            "chart_4": primary_charts.get("chart_4", empty_fig),
            "sales_chart": primary_charts.get("sales_chart", empty_fig),
            "profit_chart": primary_charts.get("profit_chart", empty_fig),
            "category_chart": primary_charts.get("category_chart", empty_fig),
            "region_chart": primary_charts.get("region_chart", empty_fig),
        }