import gradio as gr
import pandas as pd

from app.eda.loader import DataLoader
from app.executor.safe_executor import ExecutorError
from app.formatters.dashboard_formatter import DashboardFormatter
from app.formatters.verification_formatter import VerificationFormatter
from app.services.analysis_service import AnalysisService, AnalysisServiceError
from app.services.eda_service import EDAService
from app.services.planner_service import PlannerConfigurationError
from app.services.trust_score_service import TrustScoreService


def analyze_dataset(file):
    """
    Analyze uploaded dataset and return profile + visualizations.
    """
    if file is None:
        return (
            None,
            None,
            None,
            None,
            None,
        )

    result = EDAService.analyze(file.name)

    return (
        DashboardFormatter.profile_card(result["profile"]),
        result["sales_chart"],
        result["profit_chart"],
        result["category_chart"],
        result["region_chart"],
    )


def process_question(file, question, analysis_service=None):
    """
    Executes the complete self-verification analysis loop for a question:
    1. Loads dataset from file.
    2. Runs AnalysisService (Planner -> SafeExecutor -> Narrator -> ClaimExtractor -> Verification).
    3. Calculates deterministic TrustScore via TrustScoreService.
    4. Formats human-readable output, badges, and ground-truth evidence table.
    """
    if file is None:
        return (
            "⚠️ **Please upload a CSV dataset first before asking a question.**",
            "",
            "",
            pd.DataFrame(),
        )

    if not question or not str(question).strip():
        return (
            "⚠️ **Please enter a question about your dataset.**",
            "",
            "",
            pd.DataFrame(),
        )

    try:
        df = DataLoader.load_csv(file.name)
        service = analysis_service or AnalysisService()
        result = service.analyze_question(df, str(question).strip())

        trust_score = TrustScoreService.calculate(result.verification_results)

        trust_html = VerificationFormatter.trust_score_card(trust_score)
        claims_html = VerificationFormatter.claims_breakdown(
            result.claims,
            result.verification_results
        )
        evidence_df = pd.DataFrame(result.evidence)

        # Clean AI answer text
        answer_markdown = f"""### 💡 AI Answer\n\n{result.explanation}"""

        return (
            answer_markdown,
            trust_html,
            claims_html,
            evidence_df,
        )

    except PlannerConfigurationError:
        return (
            "⚠️ **Configuration Error**: `GEMINI_API_KEY` is missing or invalid. "
            "Please check your `.env` configuration file.",
            "",
            "",
            pd.DataFrame(),
        )
    except ExecutorError as err:
        return (
            f"⚠️ **Safe Execution Error**: Could not execute query on dataset. Details: {err}",
            "",
            "",
            pd.DataFrame(),
        )
    except AnalysisServiceError as err:
        return (
            f"⚠️ **Analysis Error**: {err}",
            "",
            "",
            pd.DataFrame(),
        )
    except Exception as err:
        return (
            f"⚠️ **Analysis Error**: An unexpected issue occurred while processing your query: {err}",
            "",
            "",
            pd.DataFrame(),
        )


with gr.Blocks(
    title="VeriLens AI",
    theme=gr.themes.Soft()
) as demo:

    gr.Markdown(
        """
        # 🔍 VeriLens AI

        ### Self-Verifying LLM Data Analyst
        *Autonomous exploratory analysis with deterministic ground-truth verification.*
        """
    )

    # Custom styling to ensure Plotly charts maintain full height inside Gradio flex rows
    gr.HTML(
        """
        <style>
        .block:has(.js-plotly-plot),
        .js-plotly-plot,
        .plot-container,
        .plotly {
            min-height: 420px !important;
            height: 420px !important;
        }
        </style>
        """
    )

    # -----------------------------
    # Dataset Upload & Profiling
    # -----------------------------
    with gr.Row():
        file_input = gr.File(
            label="Upload CSV Dataset",
            file_types=[".csv"]
        )

    analyze_btn = gr.Button(
        "Analyze Dataset",
        variant="primary"
    )

    profile_output = gr.HTML(
        label="Dataset Summary"
    )

    # -----------------------------
    # EDA Visualizations
    # -----------------------------
    gr.Markdown("---")
    gr.Markdown("## 📊 EDA Visualizations")

    with gr.Row():
        sales_plot = gr.Plot(show_label=False)
        profit_plot = gr.Plot(show_label=False)

    with gr.Row():
        category_plot = gr.Plot(show_label=False)
        region_plot = gr.Plot(show_label=False)

    # -----------------------------
    # AI Analysis & Self-Verification Section
    # -----------------------------
    gr.Markdown("---")
    gr.Markdown("## 🤖 AI Analysis & Self-Verification")
    gr.Markdown(
        "Ask any question about your dataset. VeriLens will reason out a plan, "
        "execute it deterministically with Pandas, narrate the findings, extract the claims, "
        "and verify every number against the ground truth."
    )

    with gr.Row():
        question_input = gr.Textbox(
            label="Ask a question about your dataset",
            placeholder="Example: Which region has the highest profit?",
            lines=2,
            scale=4
        )
        ask_btn = gr.Button(
            "Analyze & Verify",
            variant="primary",
            scale=1
        )

    # Trust Score & Answer Display
    trust_output = gr.HTML(label="Trust Score")
    answer_output = gr.Markdown(label="AI Answer")

    # Verification Details & Ground Truth
    claims_output = gr.HTML(label="Verification Breakdown")
    evidence_output = gr.DataFrame(
        label="Ground-Truth Evidence (SafeExecutor)",
        interactive=False
    )

    # -----------------------------
    # Event Wiring
    # -----------------------------
    analyze_btn.click(
        fn=analyze_dataset,
        inputs=file_input,
        outputs=[
            profile_output,
            sales_plot,
            profit_plot,
            category_plot,
            region_plot,
        ],
    )

    ask_btn.click(
        fn=process_question,
        inputs=[file_input, question_input],
        outputs=[
            answer_output,
            trust_output,
            claims_output,
            evidence_output,
        ],
    )