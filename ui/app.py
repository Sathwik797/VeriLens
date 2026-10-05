import os
import gradio as gr
import pandas as pd

from app.eda.loader import DataLoader, DataLoaderError
from app.executor.safe_executor import ExecutorError
from app.eda.safe_join_executor import SafeJoinExecutor
from app.formatters.dashboard_formatter import DashboardFormatter
from app.formatters.verification_formatter import VerificationFormatter
from app.formatters.workspace_formatter import WorkspaceFormatter
from app.services.analysis_service import AnalysisService, AnalysisServiceError
from app.services.eda_service import EDAService
from app.services.planner_service import PlannerConfigurationError
from app.services.trust_score_service import TrustScoreService


def update_join_button_visibility(file=None, files=None):
    """Determines whether to show the Execute Safe Join action box."""
    raw_input = file if file is not None else files
    if raw_input is None:
        return gr.update(visible=False)
    file_list = raw_input if isinstance(raw_input, list) else [raw_input]
    if len(file_list) < 2:
        return gr.update(visible=False)
    try:
        workspace_result = EDAService.analyze_workspace(file_list, use_gemini=False)
        recs = workspace_result.get("join_recommendations", [])
        if any(getattr(r, "safe_to_execute", False) for r in recs):
            return gr.update(visible=True)
    except Exception:
        pass
    return gr.update(visible=False)


def execute_safe_join(file=None, files=None):
    """
    Executes safe join on the uploaded datasets according to the approved
    Phase 2 recommendation. Never mutates input datasets.
    """
    raw_input = file if file is not None else files
    if raw_input is None:
        return (
            "<p style='color: #ef4444;'>No files uploaded.</p>",
            pd.DataFrame(),
        )

    file_list = raw_input if isinstance(raw_input, list) else [raw_input]
    if len(file_list) < 2:
        return (
            "<p style='color: #ef4444;'>At least two datasets are required to execute a join.</p>",
            pd.DataFrame(),
        )

    try:
        workspace_result = EDAService.analyze_workspace(file_list, use_gemini=False)
        recs = workspace_result.get("join_recommendations", [])
        safe_rec = next((r for r in recs if getattr(r, "safe_to_execute", False)), None)
        if safe_rec is None:
            return (
                "<div style='background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 14px; border-radius: 8px;'>🚫 No safe join recommendation approved for execution.</div>",
                pd.DataFrame(),
            )

        dfs = {}
        for f in file_list:
            path = f.name if hasattr(f, "name") else str(f)
            fname = getattr(f, "orig_name", None) or os.path.basename(path)
            try:
                dfs[fname] = DataLoader.load_file(path)
            except Exception:
                pass

        df_left = dfs.get(safe_rec.dataset_a)
        df_right = dfs.get(safe_rec.dataset_b)
        if df_left is None or df_right is None:
            keys = list(dfs.keys())
            if len(keys) >= 2:
                df_left = dfs[keys[0]]
                df_right = dfs[keys[1]]

        if df_left is None or df_right is None:
            return (
                "<div style='background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 14px; border-radius: 8px;'>🚫 Unable to load dataset DataFrames for execution.</div>",
                pd.DataFrame(),
            )

        output = SafeJoinExecutor.execute(
            left_df=df_left,
            right_df=df_right,
            recommendation=safe_rec,
            requested_join_type=safe_rec.recommended_join,
            user_approved=True,
            left_name=safe_rec.dataset_a,
            right_name=safe_rec.dataset_b,
        )

        card_html = WorkspaceFormatter.format_join_execution(output.result)
        preview_df = output.dataframe.head(10) if output.dataframe is not None else pd.DataFrame()
        return (card_html, preview_df)
    except Exception as err:
        return (
            f"<div style='background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 14px; border-radius: 8px;'>⚠️ Execution Error: {err}</div>",
            pd.DataFrame(),
        )



def analyze_dataset(file=None, files=None):
    """
    Analyze uploaded dataset(s) and return profile + visualizations.
    Supports single file or multi-file workspace.
    """
    raw_input = file if file is not None else files
    if raw_input is None:
        return (
            None,
            None,
            None,
            None,
            None,
        )

    # Normalize single file or list
    if not isinstance(raw_input, list):
        file_list = [raw_input]
    else:
        file_list = raw_input

    if len(file_list) == 0:
        return (None, None, None, None, None)

    # Case 1: Single file upload (existing baseline workflow)
    if len(file_list) == 1:
        single_file = file_list[0]
        path = single_file.name if hasattr(single_file, "name") else str(single_file)
        try:
            result = EDAService.analyze(path)
            sheet_name = result.get("sheet_name")

            return (
                DashboardFormatter.profile_card(result["profile"], sheet_name=sheet_name),
                result["sales_chart"],
                result["profit_chart"],
                result["category_chart"],
                result["region_chart"],
            )
        except DataLoaderError as err:
            error_html = f"""
            <div style="background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 18px; border-radius: 12px; margin: 16px 0; font-family: sans-serif;">
                <h3 style="margin-top: 0; color: #b91c1c;">⚠️ Dataset Loading Error</h3>
                <p style="margin-bottom: 0; font-size: 15px;">{err}</p>
            </div>
            """
            return (error_html, None, None, None, None)
        except Exception as err:
            error_html = f"""
            <div style="background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 18px; border-radius: 12px; margin: 16px 0; font-family: sans-serif;">
                <h3 style="margin-top: 0; color: #b91c1c;">⚠️ Dataset Analysis Error</h3>
                <p style="margin-bottom: 0; font-size: 15px;">{err}</p>
            </div>
            """
            return (error_html, None, None, None, None)

    # Case 2: Multi-file workspace upload
    try:
        workspace_result = EDAService.analyze_workspace(file_list)
        workspace_html = WorkspaceFormatter.format_workspace(
            workspace_result["profiles"],
            workspace_result["relationships"],
            join_recommendations=workspace_result.get("join_recommendations", []),
        )

        return (
            workspace_html,
            workspace_result["chart_1"],
            workspace_result["chart_2"],
            workspace_result["chart_3"],
            workspace_result["chart_4"],
        )
    except Exception as err:
        error_html = f"""
        <div style="background: #fef2f2; border: 1px solid #f87171; color: #991b1b; padding: 18px; border-radius: 12px; margin: 16px 0; font-family: sans-serif;">
            <h3 style="margin-top: 0; color: #b91c1c;">⚠️ Workspace Error</h3>
            <p style="margin-bottom: 0; font-size: 15px;">{err}</p>
        </div>
        """
        return (error_html, None, None, None, None)


def process_question(file=None, question="", analysis_service=None, files=None):
    """
    Executes the complete self-verification analysis loop for a question:
    1. Loads dataset from file.
    2. Runs AnalysisService (Planner -> SafeExecutor -> Narrator -> ClaimExtractor -> Verification).
    3. Calculates deterministic TrustScore via TrustScoreService.
    4. Formats human-readable output, badges, and ground-truth evidence table.
    """
    raw_input = file if file is not None else files
    if raw_input is None:
        return (
            "⚠️ **Please upload a CSV dataset first (or Excel .xlsx) before asking a question.**",
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

    # Normalize file input
    if isinstance(raw_input, list):
        if not raw_input:
            return (
                "⚠️ **Please upload a CSV dataset first (or Excel .xlsx) before asking a question.**",
                "",
                "",
                pd.DataFrame(),
            )
        target_file = raw_input[0]
        orig_filename = getattr(target_file, "orig_name", None) or os.path.basename(getattr(target_file, "name", str(target_file)))
        multi_note = f"\n\n*(Note: Multi-dataset workspace active. Question evaluated on primary dataset: **{orig_filename}**.)*" if len(raw_input) > 1 else ""
    else:
        target_file = raw_input
        multi_note = ""

    path = target_file.name if hasattr(target_file, "name") else str(target_file)

    try:
        df = DataLoader.load_file(path)
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
        answer_markdown = f"""### 💡 AI Answer\n\n{result.explanation}{multi_note}"""

        return (
            answer_markdown,
            trust_html,
            claims_html,
            evidence_df,
        )

    except DataLoaderError as err:
        return (
            f"⚠️ **Dataset Error**: {err}",
            "",
            "",
            pd.DataFrame(),
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
            label="Upload Dataset(s) (.csv, .xlsx)",
            file_types=[".csv", ".xlsx"],
            file_count="multiple"
        )

    analyze_btn = gr.Button(
        "Analyze Dataset",
        variant="primary"
    )

    profile_output = gr.HTML(
        label="Dataset Summary"
    )

    # -----------------------------
    # Phase 3: Safe Join Action & Execution Result
    # -----------------------------
    with gr.Column(visible=False) as join_action_box:
        gr.Markdown("### ⚡ Safe Join Action")
        gr.Markdown(
            "*A safe join relationship was detected and validated by Phase 2. "
            "Executing will create a derived joined dataset without modifying original data.*"
        )
        execute_join_btn = gr.Button(
            "⚡ Execute Safe Join",
            variant="primary",
        )
        join_execution_output = gr.HTML(
            label="Join Execution Result"
        )
        join_preview_table = gr.DataFrame(
            label="Derived Joined Dataset Preview (First 10 Rows)",
            interactive=False,
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
    ).then(
        fn=update_join_button_visibility,
        inputs=file_input,
        outputs=join_action_box,
    )

    file_input.change(
        fn=lambda: (gr.update(visible=False), "", pd.DataFrame()),
        outputs=[join_action_box, join_execution_output, join_preview_table],
    )

    execute_join_btn.click(
        fn=execute_safe_join,
        inputs=file_input,
        outputs=[
            join_execution_output,
            join_preview_table,
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