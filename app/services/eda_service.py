from app.eda.loader import DataLoader
from app.eda.profiler import DataProfiler
from app.eda.visualizer import DataVisualizer


class EDAService:

    @staticmethod
    def analyze(file_path):

        df = DataLoader.load_csv(file_path)

        profile = DataProfiler.generate_profile(df)

        charts = DataVisualizer.generate_charts(df)

        return {
            "profile": profile,
            "sales_chart": charts["sales_chart"],
            "profit_chart": charts["profit_chart"],
            "category_chart": charts["category_chart"],
            "region_chart": charts["region_chart"],
            "chart_1": charts["chart_1"],
            "chart_2": charts["chart_2"],
            "chart_3": charts["chart_3"],
            "chart_4": charts["chart_4"],
        }