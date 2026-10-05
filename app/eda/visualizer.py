import re
from typing import Dict, List, Optional
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


class DataVisualizer:
    """
    Dataset-agnostic visualization engine for exploratory data analysis (EDA).
    Automatically inspects any DataFrame schema, identifies suitable numeric,
    categorical, and datetime columns, and generates four insightful Plotly charts.
    """

    # ID/code column patterns to penalize or exclude from continuous metrics
    _ID_PATTERNS = re.compile(
        r"(^id$|_id$|id\b|code$|zip|postal|phone|ssn|index|row.?id|order.?id|cust(omer)?.?id|prod(uct)?.?id)",
        re.IGNORECASE,
    )

    # Tier 1 measure keywords (primary financial / operational KPIs)
    _TIER1_MEASURE_KEYWORDS = re.compile(
        r"(sale|revenue|profit|weekly_sales|income|turnover)",
        re.IGNORECASE,
    )

    # Tier 2 measure keywords (secondary metrics & domain measures)
    _TIER2_MEASURE_KEYWORDS = re.compile(
        r"(price|cost|amount|total|unemployment|temperature|temp|cpi|fuel_price|balance|value|score|weight|volume)",
        re.IGNORECASE,
    )

    # Tier 3 measure keywords (rates, counts, discounts, percentages)
    _TIER3_MEASURE_KEYWORDS = re.compile(
        r"(discount|qty|quantity|rate|margin|tax|fee|ratio|pct|percent)",
        re.IGNORECASE,
    )

    # Tier 1 category keywords (major groupings)
    _TIER1_CATEGORY_KEYWORDS = re.compile(
        r"(category|department|dept|region|store|type|channel)",
        re.IGNORECASE,
    )

    # Tier 2 category keywords (sub-groupings)
    _TIER2_CATEGORY_KEYWORDS = re.compile(
        r"(segment|group|class|status|division|tier|sector)",
        re.IGNORECASE,
    )

    # Tier 3 category keywords (geographic / general attributes)
    _TIER3_CATEGORY_KEYWORDS = re.compile(
        r"(country|state|city|brand|model|flag|level|gender)",
        re.IGNORECASE,
    )

    # Date / timestamp keywords
    _DATE_KEYWORDS = re.compile(
        r"([_ \t\-]|^)(date|time|timestamp|year|month|day|period)([_ \t\-]|$)",
        re.IGNORECASE,
    )

    # ---------------------------------------------------------
    # Intelligent Column Selection Helpers
    # ---------------------------------------------------------

    @classmethod
    def get_numeric_candidates(cls, df: pd.DataFrame) -> List[str]:
        """
        Returns numeric columns ranked by their likelihood of being meaningful analytical measures,
        penalizing ID, index, zip code, and degenerate constant columns.
        """
        candidates = []
        n_rows = len(df)
        if n_rows == 0:
            return []

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            n_unique = series.nunique()

            # Ignore constants
            if n_unique <= 1:
                continue

            # Penalize pure sequential index or unique IDs (e.g. 100% unique integers)
            is_id_name = bool(cls._ID_PATTERNS.search(str(col)))
            is_100_unique_int = (n_unique == n_rows) and pd.api.types.is_integer_dtype(df[col])

            if is_100_unique_int or (is_id_name and n_unique > 20):
                continue

            # Base score
            score = 10.0

            # Tiered keyword bonuses
            col_str = str(col)
            if cls._TIER1_MEASURE_KEYWORDS.search(col_str):
                score += 80.0
            elif cls._TIER2_MEASURE_KEYWORDS.search(col_str):
                score += 45.0
            elif cls._TIER3_MEASURE_KEYWORDS.search(col_str):
                score += 20.0

            # Continuous floating-point metric bonus
            if pd.api.types.is_float_dtype(df[col]):
                score += 15.0
                # Cardinality bonus for true continuous floats over discrete codes
                score += min(20.0, (n_unique / n_rows) * 40.0)

            # Penalize low-cardinality flags (e.g. 0/1 binary flags)
            if n_unique <= 2:
                score -= 30.0
            elif n_unique < 8 and pd.api.types.is_integer_dtype(df[col]):
                score -= 15.0

            candidates.append((score, col))

        # Sort descending by score
        candidates.sort(key=lambda x: x[0], reverse=True)
        return [col for _, col in candidates]

    @classmethod
    def get_categorical_candidates(cls, df: pd.DataFrame, exclude_cols: Optional[List[str]] = None) -> List[str]:
        """
        Returns categorical columns ranked by suitability for grouping (e.g. 2 to 50 unique values),
        penalizing free text, high-cardinality IDs, and constants.
        """
        exclude = set(exclude_cols or [])
        candidates = []
        n_rows = len(df)
        if n_rows == 0:
            return []

        for col in df.columns:
            if col in exclude:
                continue

            # Date columns should not be used as standard categories
            if cls._DATE_KEYWORDS.search(str(col)):
                continue

            series = df[col].dropna()
            n_unique = series.nunique()

            # Discard constants or extreme cardinality
            if n_unique <= 1:
                continue

            # Discard unique IDs / free text / descriptions
            if n_rows > 10 and (n_unique / n_rows > 0.6 or n_unique > 60):
                continue

            # Discard if column name matches ID patterns and cardinality is high
            if cls._ID_PATTERNS.search(str(col)) and n_unique > 15:
                continue

            score = 10.0

            # Ideal category range: between 2 and 25 unique items
            if 2 <= n_unique <= 25:
                score += 35.0
            elif 26 <= n_unique <= 50:
                score += 15.0

            # Tiered keyword bonuses
            col_str = str(col)
            if re.search(r"(category|department|dept)", col_str, re.IGNORECASE):
                score += 55.0
            elif cls._TIER1_CATEGORY_KEYWORDS.search(col_str):
                score += 50.0
            elif cls._TIER2_CATEGORY_KEYWORDS.search(col_str):
                score += 30.0
            elif cls._TIER3_CATEGORY_KEYWORDS.search(col_str):
                score += 15.0

            # String/object/category dtype bonus
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
                score += 20.0

            candidates.append((score, col))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [col for _, col in candidates]

    @classmethod
    def get_datetime_candidates(cls, df: pd.DataFrame) -> List[str]:
        """
        Detects datetime columns or string/object columns containing date information.
        """
        candidates = []

        for col in df.columns:
            # 1. True datetime dtype
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                candidates.append((100.0, col))
                continue

            # Exclude flags, binary columns, and non-temporal low-cardinality integers
            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 5:
                continue

            # 2. String/object columns with date-like names
            if cls._DATE_KEYWORDS.search(str(col)):
                # Sample non-null values to verify they parse as dates
                sample = df[col].dropna().head(15)
                if not sample.empty:
                    try:
                        parsed = pd.to_datetime(sample, format="mixed", errors="coerce")
                        if parsed.notna().sum() >= len(sample) * 0.5:
                            candidates.append((80.0, col))
                    except Exception:
                        pass

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [col for _, col in candidates]

    # ---------------------------------------------------------
    # Chart Generators
    # ---------------------------------------------------------

    @classmethod
    def create_distribution_chart(cls, df: pd.DataFrame, col: str) -> go.Figure:
        """Creates a clean histogram for a numeric or discrete column."""
        fig = px.histogram(
            df,
            x=col,
            nbins=30,
            title=f"{col} Distribution",
            template="plotly_white",
        )
        fig.update_layout(
            bargap=0.1,
            margin=dict(l=40, r=40, t=50, b=40),
            title_font=dict(size=15),
        )
        return fig

    @classmethod
    def create_category_bar_chart(cls, df: pd.DataFrame, cat_col: str, num_col: str) -> go.Figure:
        """Creates an aggregated bar chart of num_col grouped by cat_col."""
        agg = (
            df.groupby(cat_col, as_index=False)[num_col]
            .sum()
            .sort_values(by=num_col, ascending=False)
            .head(15)
        )

        fig = px.bar(
            agg,
            x=cat_col,
            y=num_col,
            title=f"{num_col} by {cat_col}",
            template="plotly_white",
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            title_font=dict(size=15),
        )
        return fig

    @classmethod
    def create_time_trend_chart(cls, df: pd.DataFrame, date_col: str, num_col: str) -> go.Figure:
        """Creates a time series trend chart for a numeric column."""
        temp_df = df[[date_col, num_col]].dropna().copy()
        temp_df["_parsed_date"] = pd.to_datetime(temp_df[date_col], format="mixed", errors="coerce")
        temp_df = temp_df.dropna(subset=["_parsed_date"])

        if temp_df.empty:
            return cls.create_empty_chart("No valid date records for trend")

        # Sort chronologically
        temp_df = temp_df.sort_values("_parsed_date")

        # If more than 60 points, aggregate by period (Month or Week) for clean visualization
        n_points = temp_df["_parsed_date"].dt.date.nunique()
        if n_points > 90:
            temp_df["_period"] = temp_df["_parsed_date"].dt.to_period("M").dt.to_timestamp()
            trend = temp_df.groupby("_period", as_index=False)[num_col].sum()
            x_col = "_period"
        elif n_points > 40:
            temp_df["_period"] = temp_df["_parsed_date"].dt.to_period("W").dt.to_timestamp()
            trend = temp_df.groupby("_period", as_index=False)[num_col].sum()
            x_col = "_period"
        else:
            trend = temp_df.groupby("_parsed_date", as_index=False)[num_col].sum()
            x_col = "_parsed_date"

        fig = px.line(
            trend,
            x=x_col,
            y=num_col,
            title=f"{num_col} Trend Over Time",
            template="plotly_white",
            markers=True if len(trend) <= 40 else False,
        )
        fig.update_layout(
            xaxis_title=date_col,
            margin=dict(l=40, r=40, t=50, b=40),
            title_font=dict(size=15),
        )
        return fig

    @classmethod
    def create_category_frequency_chart(cls, df: pd.DataFrame, cat_col: str) -> go.Figure:
        """Creates a count bar chart for a categorical column."""
        counts = df[cat_col].value_counts().head(15).reset_index()
        counts.columns = [cat_col, "Count"]

        fig = px.bar(
            counts,
            x=cat_col,
            y="Count",
            title=f"{cat_col} Frequency",
            template="plotly_white",
        )
        fig.update_layout(
            margin=dict(l=40, r=40, t=50, b=40),
            title_font=dict(size=15),
        )
        return fig

    @classmethod
    def create_empty_chart(cls, title: str = "Insufficient Data") -> go.Figure:
        """Returns a gracefully rendered empty figure with a clear notice."""
        fig = go.Figure()
        fig.update_layout(
            title=title,
            template="plotly_white",
            margin=dict(l=40, r=40, t=50, b=40),
            annotations=[
                dict(
                    text="No suitable columns available for this chart.",
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="gray"),
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        return fig

    # ---------------------------------------------------------
    # Main Generic Chart Generation Pipeline
    # ---------------------------------------------------------

    @classmethod
    def generate_charts(cls, df: pd.DataFrame) -> Dict[str, go.Figure]:
        """
        Dynamically analyzes the DataFrame and returns four cohesive, dataset-agnostic charts.
        Returns a dictionary with both generic ('chart_1'..'chart_4') and backward-compatible
        ('sales_chart', 'profit_chart', 'category_chart', 'region_chart') keys.
        """
        if df is None or df.empty:
            empty = cls.create_empty_chart("No Data Available")
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

        numeric_cols = cls.get_numeric_candidates(df)
        cat_cols = cls.get_categorical_candidates(df)
        date_cols = cls.get_datetime_candidates(df)

        primary_num = numeric_cols[0] if len(numeric_cols) > 0 else None
        secondary_num = numeric_cols[1] if len(numeric_cols) > 1 else None

        primary_cat = cat_cols[0] if len(cat_cols) > 0 else None
        secondary_cat = cat_cols[1] if len(cat_cols) > 1 else None

        primary_date = date_cols[0] if len(date_cols) > 0 else None

        # -----------------------------------------------------
        # Chart 1: Primary Numeric Distribution
        # -----------------------------------------------------
        if primary_num:
            chart_1 = cls.create_distribution_chart(df, primary_num)
        elif primary_cat:
            chart_1 = cls.create_category_frequency_chart(df, primary_cat)
        else:
            chart_1 = cls.create_empty_chart("Distribution Chart (No suitable columns)")

        # -----------------------------------------------------
        # Chart 2: Secondary Numeric Distribution or Category Breakdown
        # -----------------------------------------------------
        if secondary_num:
            chart_2 = cls.create_distribution_chart(df, secondary_num)
        elif primary_cat and primary_num:
            # If only 1 numeric column exists, show categorical frequency
            chart_2 = cls.create_category_frequency_chart(df, primary_cat)
        elif secondary_cat:
            chart_2 = cls.create_category_frequency_chart(df, secondary_cat)
        elif primary_num:
            # Only 1 numeric column exists in total, show box plot for spread
            fig = px.box(df, y=primary_num, title=f"{primary_num} Spread (Box Plot)", template="plotly_white")
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15))
            chart_2 = fig
        else:
            chart_2 = cls.create_empty_chart("Secondary Distribution (No suitable columns)")

        # -----------------------------------------------------
        # Chart 3: Categorical-vs-Numeric Aggregated Bar Chart
        # -----------------------------------------------------
        if primary_cat and primary_num:
            chart_3 = cls.create_category_bar_chart(df, primary_cat, primary_num)
        elif primary_cat:
            chart_3 = cls.create_category_frequency_chart(df, primary_cat)
        elif primary_num and secondary_num:
            fig = px.scatter(
                df, x=primary_num, y=secondary_num,
                title=f"{secondary_num} vs {primary_num}",
                template="plotly_white",
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15))
            chart_3 = fig
        else:
            chart_3 = cls.create_empty_chart("Category Aggregation (No suitable columns)")

        # -----------------------------------------------------
        # Chart 4: Time Trend or Secondary Categorical Aggregation
        # -----------------------------------------------------
        if primary_date and primary_num:
            chart_4 = cls.create_time_trend_chart(df, primary_date, primary_num)
        elif secondary_cat and (secondary_num or primary_num):
            metric = secondary_num if secondary_num else primary_num
            chart_4 = cls.create_category_bar_chart(df, secondary_cat, metric)
        elif primary_cat and secondary_num:
            chart_4 = cls.create_category_bar_chart(df, primary_cat, secondary_num)
        elif primary_num and secondary_num:
            fig = px.line(
                df.head(100), y=primary_num,
                title=f"{primary_num} Sequence",
                template="plotly_white",
            )
            fig.update_layout(margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15))
            chart_4 = fig
        else:
            chart_4 = cls.create_empty_chart("Trend / Secondary Chart (No suitable columns)")

        return {
            "chart_1": chart_1,
            "chart_2": chart_2,
            "chart_3": chart_3,
            "chart_4": chart_4,
            "sales_chart": chart_1,
            "profit_chart": chart_2,
            "category_chart": chart_3,
            "region_chart": chart_4,
        }

    # ---------------------------------------------------------
    # Backward Compatibility Methods for Superstore or Existing Code
    # ---------------------------------------------------------

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