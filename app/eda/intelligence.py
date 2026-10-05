import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dotenv import load_dotenv

from app.models.visualization_candidate import VisualizationCandidate, VisualizationType

load_dotenv()
logger = logging.getLogger(__name__)

# ID/code column patterns
_ID_PATTERNS = re.compile(
    r"(^id$|_id$|id\b|code$|zip|postal|phone|ssn|index|row.?id|order.?id|cust(omer)?.?id|prod(uct)?.?id)",
    re.IGNORECASE,
)

# Date / timestamp keywords (word/boundary matching)
_DATE_KEYWORDS = re.compile(
    r"([_ \t\-]|^)(date|time|timestamp|year|month|day|period)([_ \t\-]|$)",
    re.IGNORECASE,
)

# Primary measure keywords
_TIER1_MEASURE_KEYWORDS = re.compile(
    r"(sale|revenue|profit|weekly_sales|income|turnover)",
    re.IGNORECASE,
)

# Secondary measure keywords
_TIER2_MEASURE_KEYWORDS = re.compile(
    r"(price|cost|amount|total|unemployment|temperature|temp|cpi|fuel_price|balance|value|score|weight|volume)",
    re.IGNORECASE,
)

# Rate / non-additive measure keywords
_RATE_KEYWORDS = re.compile(
    r"(rate|pct|percent|ratio|discount|margin|temperature|temp|cpi|unemployment)",
    re.IGNORECASE,
)


class CandidateGenerator:
    """
    Generates deterministic visualization candidates from any pandas DataFrame schema.
    """

    @staticmethod
    def get_numeric_columns(df: pd.DataFrame) -> List[str]:
        """Returns non-ID, non-constant numeric columns ranked by measure likelihood."""
        candidates = []
        n_rows = len(df)
        if n_rows == 0:
            return []

        for col in df.columns:
            if not pd.api.types.is_numeric_dtype(df[col]):
                continue

            series = df[col].dropna()
            n_unique = series.nunique()

            if n_unique <= 1:
                continue

            # Exclude 100% unique sequence integers or ID-like columns
            is_id_name = bool(_ID_PATTERNS.search(str(col)))
            is_unique_seq = (n_unique == n_rows) and pd.api.types.is_integer_dtype(df[col])
            if is_unique_seq or (is_id_name and n_unique > 20):
                continue

            score = 10.0
            col_str = str(col)
            if _TIER1_MEASURE_KEYWORDS.search(col_str):
                score += 80.0
            elif _TIER2_MEASURE_KEYWORDS.search(col_str):
                score += 45.0

            if pd.api.types.is_float_dtype(df[col]):
                score += 15.0
                score += min(20.0, (n_unique / n_rows) * 40.0)

            # Penalize low-cardinality binary flags (0/1)
            if n_unique <= 2:
                score -= 30.0
            elif n_unique < 8 and pd.api.types.is_integer_dtype(df[col]):
                score -= 15.0

            candidates.append((score, col))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [col for _, col in candidates]

    @staticmethod
    def get_categorical_columns(df: pd.DataFrame) -> List[str]:
        """Returns categorical columns suitable for grouping (2 <= nunique <= 40)."""
        candidates = []
        n_rows = len(df)
        if n_rows == 0:
            return []

        for col in df.columns:
            col_str = str(col)
            # Skip dates
            if _DATE_KEYWORDS.search(col_str):
                continue

            # Continuous floats should never be categories
            if pd.api.types.is_float_dtype(df[col]):
                continue

            series = df[col].dropna()
            n_unique = series.nunique()

            # Reject constants or extreme cardinality
            if n_unique <= 1:
                continue
            if n_rows > 10 and (n_unique / n_rows > 0.6 or n_unique > 60):
                continue
            if _ID_PATTERNS.search(col_str) and n_unique > 15:
                continue

            # If integer, only allow low cardinality codes (e.g. Holiday_Flag with 0/1 or Store with few IDs)
            if pd.api.types.is_integer_dtype(df[col]) and n_unique > 15:
                continue

            score = 10.0
            if 2 <= n_unique <= 8:
                score += 45.0  # Perfect for pie/donut and bar
            elif 9 <= n_unique <= 25:
                score += 35.0
            elif 26 <= n_unique <= 50:
                score += 15.0

            if re.search(r"(category|department|dept)", col_str, re.IGNORECASE):
                score += 55.0
            elif re.search(r"(region|store|type|channel|segment|group)", col_str, re.IGNORECASE):
                score += 45.0

            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
                score += 20.0

            candidates.append((score, col))

        candidates.sort(key=lambda x: x[0], reverse=True)
        return [col for _, col in candidates]

    @staticmethod
    def get_datetime_columns(df: pd.DataFrame) -> List[str]:
        """Returns verified datetime columns or parseable string columns."""
        candidates = []
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                candidates.append((100.0, col))
                continue

            if pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 5:
                continue

            if _DATE_KEYWORDS.search(str(col)):
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

    @classmethod
    def generate_candidates(cls, df: pd.DataFrame) -> List[VisualizationCandidate]:
        """Generates all valid visualization candidates across supported chart types."""
        candidates: List[VisualizationCandidate] = []
        if df is None or df.empty:
            return candidates

        num_cols = cls.get_numeric_columns(df)
        cat_cols = cls.get_categorical_columns(df)
        date_cols = cls.get_datetime_columns(df)

        # -----------------------------------------------------
        # 1. Bar Chart Candidates (Categorical + Numeric)
        # -----------------------------------------------------
        for cat in cat_cols[:3]:
            cat_nunique = df[cat].dropna().nunique()
            if 2 <= cat_nunique <= 40:
                for num in num_cols[:3]:
                    if cat == num:
                        continue
                    is_rate = bool(_RATE_KEYWORDS.search(str(num)))
                    agg: Any = "mean" if is_rate else "sum"
                    title = f"Average {num} by {cat}" if agg == "mean" else f"{num} by {cat}"
                    score = 80.0 + (10.0 if _TIER1_MEASURE_KEYWORDS.search(str(num)) else 0.0)
                    candidates.append(
                        VisualizationCandidate(
                            candidate_id=f"bar_{cat}_{num}",
                            chart_type="bar",
                            primary_column=cat,
                            secondary_column=num,
                            aggregation=agg,
                            title=title,
                            description=f"Aggregated {agg} of {num} across {cat}",
                            suitability_score=min(score, 98.0),
                            metadata={"cardinality": cat_nunique},
                        )
                    )

        # -----------------------------------------------------
        # 2. Pie / Donut Chart Candidates (Clean 2-8 Categories + Positive Metric)
        # -----------------------------------------------------
        for cat in cat_cols[:3]:
            cat_nunique = df[cat].dropna().nunique()
            # Strictly 2 to 8 categories for pie/donut readability
            if 2 <= cat_nunique <= 8:
                for num in num_cols[:2]:
                    if cat == num:
                        continue
                    # Exclude rates, CPI, temperature from pie charts
                    if _RATE_KEYWORDS.search(str(num)):
                        continue
                    # Must be additive and mostly non-negative
                    val_sum = df[num].dropna().sum()
                    if val_sum <= 0:
                        continue
                    neg_ratio = (df[num] < 0).mean()
                    if neg_ratio > 0.05:
                        continue

                    score = 88.0 + (10.0 if _TIER1_MEASURE_KEYWORDS.search(str(num)) else 0.0)
                    candidates.append(
                        VisualizationCandidate(
                            candidate_id=f"donut_{cat}_{num}",
                            chart_type="donut",
                            primary_column=cat,
                            secondary_column=num,
                            aggregation="sum",
                            title=f"{num} Share by {cat}",
                            description=f"Proportional breakdown of {num} across {cat_nunique} categories in {cat}",
                            suitability_score=min(score, 99.0),
                            metadata={"cardinality": cat_nunique},
                        )
                    )

        # -----------------------------------------------------
        # 3. Line / Trend Candidates (Date + Numeric)
        # -----------------------------------------------------
        for dt_col in date_cols[:2]:
            for num in num_cols[:2]:
                score = 85.0 + (10.0 if _TIER1_MEASURE_KEYWORDS.search(str(num)) else 0.0)
                candidates.append(
                    VisualizationCandidate(
                        candidate_id=f"line_{dt_col}_{num}",
                        chart_type="line",
                        primary_column=dt_col,
                        secondary_column=num,
                        aggregation="sum" if not _RATE_KEYWORDS.search(str(num)) else "mean",
                        title=f"{num} Trend Over Time",
                        description=f"Chronological trajectory of {num} along {dt_col}",
                        suitability_score=min(score, 97.0),
                    )
                )

        # -----------------------------------------------------
        # 4. Scatter Plot Candidates (Numeric vs Numeric)
        # -----------------------------------------------------
        if len(num_cols) >= 2:
            for i in range(min(3, len(num_cols))):
                for j in range(i + 1, min(4, len(num_cols))):
                    c1, c2 = num_cols[i], num_cols[j]
                    u1 = df[c1].dropna().nunique()
                    u2 = df[c2].dropna().nunique()
                    # Quality rule: both must have continuous variation
                    if u1 > 10 and u2 > 10:
                        score = 75.0 + (10.0 if _TIER1_MEASURE_KEYWORDS.search(str(c1)) or _TIER1_MEASURE_KEYWORDS.search(str(c2)) else 0.0)
                        candidates.append(
                            VisualizationCandidate(
                                candidate_id=f"scatter_{c1}_{c2}",
                                chart_type="scatter",
                                primary_column=c1,
                                secondary_column=c2,
                                title=f"{c2} vs {c1}",
                                description=f"Correlation pattern between {c2} and {c1}",
                                suitability_score=min(score, 92.0),
                            )
                        )

        # -----------------------------------------------------
        # 5. Histogram Candidates (Univariate Measure Distribution)
        # -----------------------------------------------------
        for num in num_cols[:3]:
            u = df[num].dropna().nunique()
            if u > 5:
                score = 72.0 + (10.0 if _TIER1_MEASURE_KEYWORDS.search(str(num)) else 0.0)
                candidates.append(
                    VisualizationCandidate(
                        candidate_id=f"histogram_{num}",
                        chart_type="histogram",
                        primary_column=num,
                        title=f"{num} Distribution",
                        description=f"Frequency distribution and spread of {num}",
                        suitability_score=min(score, 90.0),
                    )
                )

        return candidates


class VisualizationRanker:
    """
    Selects up to four diverse, high-insight visualization candidates.
    Uses Gemini for semantic ranking when available, falling back cleanly to
    deterministic diversity selection.
    """

    @classmethod
    def select_diverse_candidates(cls, candidates: List[VisualizationCandidate], limit: int = 4) -> List[VisualizationCandidate]:
        """
        Deterministic diversity selection:
        Picks the highest scoring candidate from each distinct chart type first,
        then fills any remaining slots with the next highest scoring candidates.
        """
        if not candidates:
            return []

        # Sort all candidates by suitability score descending
        sorted_candidates = sorted(candidates, key=lambda c: c.suitability_score, reverse=True)

        selected: List[VisualizationCandidate] = []
        selected_types = set()
        selected_pairs = set()

        # Preferred chart diversity order
        preferred_types = ["bar", "donut", "line", "scatter", "histogram", "pie"]

        # First pass: one candidate per preferred chart type
        for p_type in preferred_types:
            matching = [
                c for c in sorted_candidates 
                if (c.chart_type == p_type or (p_type == "donut" and c.chart_type in ["pie", "donut"]))
                and c.chart_type not in selected_types
            ]
            if not matching:
                continue

            # Prioritize matching candidate with an unused column pair
            chosen = None
            for c in matching:
                pair = (c.primary_column, c.secondary_column)
                if pair not in selected_pairs:
                    chosen = c
                    break
            # If all candidates share a pair, still pick the best candidate of this chart type
            if not chosen:
                chosen = matching[0]

            selected.append(chosen)
            selected_types.add(chosen.chart_type)
            if chosen.chart_type in ["pie", "donut"]:
                selected_types.add("pie")
                selected_types.add("donut")
            selected_pairs.add((chosen.primary_column, chosen.secondary_column))

            if len(selected) >= limit:
                break

        # Second pass: fill any remaining slots with highest scoring remaining candidates
        if len(selected) < limit:
            selected_ids = {c.candidate_id for c in selected}
            for c in sorted_candidates:
                if c.candidate_id not in selected_ids:
                    pair = (c.primary_column, c.secondary_column)
                    # Prefer different column pairs
                    if pair not in selected_pairs or len(sorted_candidates) <= limit:
                        selected.append(c)
                        selected_ids.add(c.candidate_id)
                        selected_pairs.add(pair)
                if len(selected) >= limit:
                    break

        return selected[:limit]

    @classmethod
    def rank_with_gemini(
        cls,
        candidates: List[VisualizationCandidate],
        column_names: List[str],
        data_types: Dict[str, str],
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        limit: int = 4,
    ) -> List[VisualizationCandidate]:
        """
        Uses Gemini to select up to limit candidates maximizing visual diversity and analytical insight.
        Falls back to deterministic selection if Gemini is unavailable, errors, or returns invalid JSON.
        """
        if not candidates:
            return []

        # If 4 or fewer candidates exist, diversity selection handles it immediately
        if len(candidates) <= limit:
            return cls.select_diverse_candidates(candidates, limit=limit)

        key = api_key or os.getenv("GEMINI_API_KEY")
        if not key:
            logger.info("GEMINI_API_KEY not configured. Using deterministic diversity selection.")
            return cls.select_diverse_candidates(candidates, limit=limit)

        try:
            import google.generativeai as genai
            genai.configure(api_key=key)
            llm = genai.GenerativeModel(
                model_name=model_name or os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
            )

            # Compact representation of candidate metadata (no raw dataframe rows!)
            compact_candidates = [
                {
                    "candidate_id": c.candidate_id,
                    "chart_type": c.chart_type,
                    "primary_column": c.primary_column,
                    "secondary_column": c.secondary_column,
                    "title": c.title,
                    "description": c.description,
                    "suitability_score": c.suitability_score,
                }
                for c in candidates
            ]

            prompt = f"""You are the visualization intelligence component of VeriLens AI.
Select the top {limit} visualization candidates that maximize BOTH visual diversity and analytical insight.

Goal:
- Prefer a diverse mix of chart types (e.g. 1 bar, 1 donut/pie, 1 line/trend, 1 scatter/histogram) if supported by the dataset.
- Do NOT pick multiple charts that show identical information.
- Select strictly from the candidate IDs provided.

Dataset Columns: {column_names}
Column Data Types: {data_types}

Available Visualization Candidates:
{json.dumps(compact_candidates, indent=2)}

Respond ONLY with valid JSON matching this exact structure:
{{
  "selected_candidates": [
    {{"candidate_id": "<id>", "rank": 1}},
    {{"candidate_id": "<id>", "rank": 2}},
    {{"candidate_id": "<id>", "rank": 3}},
    {{"candidate_id": "<id>", "rank": 4}}
  ]
}}
"""
            response = llm.generate_content(
                prompt,
                generation_config=dict(temperature=0.0, response_mime_type="application/json")
            )

            text = response.text.strip()
            data = json.loads(text)
            selected_ids = [item["candidate_id"] for item in data.get("selected_candidates", [])]

            candidate_map = {c.candidate_id: c for c in candidates}
            selected = [candidate_map[cid] for cid in selected_ids if cid in candidate_map]

            # If Gemini returned at least 2 valid candidates, fill any missing up to limit
            if len(selected) >= 2:
                if len(selected) < limit:
                    existing_ids = {c.candidate_id for c in selected}
                    deterministic_fill = cls.select_diverse_candidates(candidates, limit=limit)
                    for c in deterministic_fill:
                        if c.candidate_id not in existing_ids:
                            selected.append(c)
                            existing_ids.add(c.candidate_id)
                        if len(selected) >= limit:
                            break
                return selected[:limit]
            else:
                logger.warning("Gemini returned insufficient valid candidates. Falling back to deterministic selection.")
                return cls.select_diverse_candidates(candidates, limit=limit)

        except Exception as e:
            logger.warning(f"Gemini ranking failed or timed out ({e}). Falling back to deterministic selection.")
            return cls.select_diverse_candidates(candidates, limit=limit)


class ChartRenderer:
    """
    Renders VisualizationCandidate specifications into interactive Plotly figures
    using pure deterministic Pandas calculations.
    """

    @classmethod
    def render(cls, df: pd.DataFrame, candidate: VisualizationCandidate) -> go.Figure:
        """Computes aggregation with Pandas and renders the corresponding Plotly Figure."""
        c_type = candidate.chart_type
        p_col = candidate.primary_column
        s_col = candidate.secondary_column

        if c_type == "bar":
            return cls._render_bar(df, p_col, s_col, candidate.aggregation or "sum", candidate.title)
        elif c_type in ["pie", "donut"]:
            return cls._render_pie(df, p_col, s_col, candidate.title, hole=0.4 if c_type == "donut" else 0.0)
        elif c_type == "line":
            return cls._render_line(df, p_col, s_col, candidate.title)
        elif c_type == "scatter":
            return cls._render_scatter(df, p_col, s_col, candidate.title)
        elif c_type == "histogram":
            return cls._render_histogram(df, p_col, candidate.title)
        else:
            return cls.create_empty_chart(candidate.title)

    @classmethod
    def _render_bar(cls, df: pd.DataFrame, cat_col: str, num_col: str, agg: str, title: str) -> go.Figure:
        if cat_col == num_col or cat_col not in df.columns or num_col not in df.columns:
            return cls.create_empty_chart(title)
        clean_df = df[[cat_col, num_col]].dropna()
        if clean_df.empty:
            return cls.create_empty_chart(title)

        if agg == "mean":
            grouped = clean_df.groupby(cat_col, as_index=False)[num_col].mean()
        else:
            grouped = clean_df.groupby(cat_col, as_index=False)[num_col].sum()

        grouped = grouped.sort_values(by=num_col, ascending=False).head(15)

        fig = px.bar(grouped, x=cat_col, y=num_col, title=title, template="plotly_white")
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15))
        return fig

    @classmethod
    def _render_pie(cls, df: pd.DataFrame, cat_col: str, num_col: str, title: str, hole: float = 0.4) -> go.Figure:
        if cat_col == num_col or cat_col not in df.columns or num_col not in df.columns:
            return cls.create_empty_chart(title)
        clean_df = df[[cat_col, num_col]].dropna()
        clean_df = clean_df[clean_df[num_col] > 0]
        if clean_df.empty:
            return cls.create_empty_chart(title)

        grouped = clean_df.groupby(cat_col, as_index=False)[num_col].sum()
        grouped = grouped.sort_values(by=num_col, ascending=False).head(8)

        fig = px.pie(
            grouped,
            names=cat_col,
            values=num_col,
            title=title,
            hole=hole,
            template="plotly_white",
        )
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15))
        return fig

    @classmethod
    def _render_line(cls, df: pd.DataFrame, date_col: str, num_col: str, title: str) -> go.Figure:
        if date_col == num_col or date_col not in df.columns or num_col not in df.columns:
            return cls.create_empty_chart(title)
        clean_df = df[[date_col, num_col]].dropna().copy()
        clean_df["_parsed_date"] = pd.to_datetime(clean_df[date_col], format="mixed", errors="coerce")
        clean_df = clean_df.dropna(subset=["_parsed_date"]).sort_values("_parsed_date")

        if clean_df.empty:
            return cls.create_empty_chart(title)

        n_unique_dates = clean_df["_parsed_date"].dt.date.nunique()
        if n_unique_dates > 90:
            clean_df["_period"] = clean_df["_parsed_date"].dt.to_period("M").dt.to_timestamp()
            trend = clean_df.groupby("_period", as_index=False)[num_col].sum()
            x_col = "_period"
        elif n_unique_dates > 40:
            clean_df["_period"] = clean_df["_parsed_date"].dt.to_period("W").dt.to_timestamp()
            trend = clean_df.groupby("_period", as_index=False)[num_col].sum()
            x_col = "_period"
        else:
            trend = clean_df.groupby("_parsed_date", as_index=False)[num_col].sum()
            x_col = "_parsed_date"

        fig = px.line(
            trend,
            x=x_col,
            y=num_col,
            title=title,
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
    def _render_scatter(cls, df: pd.DataFrame, num1: str, num2: str, title: str) -> go.Figure:
        if num1 == num2 or num1 not in df.columns or num2 not in df.columns:
            return cls.create_empty_chart(title)
        clean_df = df[[num1, num2]].dropna()
        if clean_df.empty:
            return cls.create_empty_chart(title)

        # Sample if huge to maintain snappy browser response
        plot_df = clean_df.sample(min(len(clean_df), 1500), random_state=42) if len(clean_df) > 1500 else clean_df

        fig = px.scatter(
            plot_df,
            x=num1,
            y=num2,
            title=title,
            template="plotly_white",
            opacity=0.6,
        )
        fig.update_layout(margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15))
        return fig

    @classmethod
    def _render_histogram(cls, df: pd.DataFrame, num_col: str, title: str) -> go.Figure:
        if num_col not in df.columns:
            return cls.create_empty_chart(title)
        clean_df = df[[num_col]].dropna()
        if clean_df.empty:
            return cls.create_empty_chart(title)

        fig = px.histogram(clean_df, x=num_col, nbins=30, title=title, template="plotly_white")
        fig.update_layout(bargap=0.1, margin=dict(l=40, r=40, t=50, b=40), title_font=dict(size=15))
        return fig

    @classmethod
    def create_empty_chart(cls, title: str = "Insufficient Data") -> go.Figure:
        """Returns a gracefully rendered empty figure with clear user notice."""
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
