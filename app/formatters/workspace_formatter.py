from typing import Any, List, Optional
from app.models.dataset_profile import DatasetProfile
from app.models.dataset_relationship import DatasetRelationship
from app.eda.column_normalizer import ColumnNormalizer


class WorkspaceFormatter:
    """
    Renders clean, styled HTML presentation for multi-dataset workspaces
    and pairwise dataset relationships.
    """

    @classmethod
    def format_workspace(
        cls,
        profiles: List[DatasetProfile],
        relationships: List[DatasetRelationship],
        join_recommendations: Optional[List[Any]] = None,
    ) -> str:
        """Generates comprehensive Workspace, Compatibility, and Join Intelligence summary HTML."""
        if not profiles:
            return ""

        # Section 1: Uploaded Datasets Grid
        dataset_cards = []
        for p in profiles:
            name = p.filename or "Unnamed Dataset"
            # Ensure index-like columns are never displayed as business keys
            clean_keys = [k for k in p.candidate_keys if not ColumnNormalizer.is_index_like(k)]
            keys_str = f"🔑 Key: <code>{', '.join(clean_keys[:2])}</code>" if clean_keys else ""
            sheet_str = f" (Sheet: <i>{p.sheet_name}</i>)" if p.sheet_name else ""

            card = f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 14px 18px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="font-weight: 600; font-size: 15px; color: #1e293b; margin-bottom: 4px;">
                    📄 {name}{sheet_str}
                </div>
                <div style="font-size: 13px; color: #64748b; margin-bottom: 6px;">
                    <strong>{p.rows:,}</strong> rows · <strong>{p.columns}</strong> columns · {p.memory_usage_mb} MB
                </div>
                <div style="font-size: 12px; color: #475569;">
                    {keys_str}
                </div>
            </div>
            """
            dataset_cards.append(card)

        datasets_html = "\n".join(dataset_cards)

        # Section 2: Dataset Relationships
        rel_cards = []
        badge_styles = {
            "COMPATIBLE": "background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;",
            "RELATED": "background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;",
            "UNRELATED": "background: #f8fafc; color: #475569; border: 1px solid #cbd5e1;",
        }
        badge_icons = {
            "COMPATIBLE": "🟢",
            "RELATED": "🔗",
            "UNRELATED": "⚪",
        }

        for r in relationships:
            style = badge_styles.get(r.classification, badge_styles["UNRELATED"])
            icon = badge_icons.get(r.classification, "⚪")

            evidence_items = "".join(f"<li style='margin-bottom: 3px;'>{ev}</li>" for ev in r.evidence)
            rec_html = f"<div style='margin-top: 8px; font-size: 12px; color: #334155; font-style: italic;'>💡 {r.recommendation}</div>" if r.recommendation else ""

            card = f"""
            <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 16px; margin-bottom: 12px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; flex-wrap: wrap; gap: 8px;">
                    <div style="font-size: 15px; font-weight: 600; color: #0f172a;">
                        📄 {r.dataset_a} &nbsp; ↔ &nbsp; 📄 {r.dataset_b}
                    </div>
                    <span style="{style} font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 16px; letter-spacing: 0.5px;">
                        {icon} {r.classification} &nbsp;({int(r.confidence * 100)}%)
                    </span>
                </div>
                <ul style="margin: 0; padding-left: 20px; font-size: 13px; color: #475569;">
                    {evidence_items}
                </ul>
                {rec_html}
            </div>
            """
            rel_cards.append(card)

        relationships_html = "\n".join(rel_cards) if rel_cards else "<p style='color: #64748b;'>No pairwise relationships detected.</p>"

        # Section 3: Safe Join Intelligence Recommendations
        join_section_html = ""
        if join_recommendations:
            join_cards = []
            conf_styles = {
                "HIGH": "background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;",
                "MODERATE": "background: #eef2ff; color: #3730a3; border: 1px solid #c7d2fe;",
                "LOW": "background: #fffbeb; color: #92400e; border: 1px solid #fde68a;",
                "VERY LOW": "background: #f1f5f9; color: #475569; border: 1px solid #cbd5e1;",
            }

            for j in join_recommendations:
                c_style = conf_styles.get(j.confidence_level, conf_styles["VERY LOW"])
                card_badge = (
                    f"<span style='background: #f0fdf4; color: #15803d; border: 1px solid #bbf7d0; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 12px;'>"
                    f"✨ {j.recommended_join.replace('_', ' ')}</span>"
                    if j.safe_to_execute
                    else "<span style='background: #fef2f2; color: #b91c1c; border: 1px solid #fecaca; font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 12px;'>❌ NO SAFE JOIN</span>"
                )

                cardinality_label = j.cardinality.replace("_", "-")
                evidence_list = "".join(f"<li style='margin-bottom: 2px;'>{ev}</li>" for ev in j.evidence)

                warnings_html = ""
                if j.warnings:
                    warn_items = "".join(f"<li style='margin-bottom: 2px;'>{w}</li>" for w in j.warnings)
                    warnings_html = f"""
                    <div style="background: #fffbeb; border: 1px solid #fef08a; border-radius: 8px; padding: 10px 14px; margin-top: 10px;">
                        <div style="font-weight: 600; font-size: 12px; color: #854d0e; margin-bottom: 4px;">⚠️ Warnings & Data Discrepancies:</div>
                        <ul style="margin: 0; padding-left: 18px; font-size: 12px; color: #713f12;">
                            {warn_items}
                        </ul>
                    </div>
                    """

                candidates_html = ""
                if len(j.all_candidates) > 1:
                    cand_rows = []
                    for idx, c in enumerate(j.all_candidates, 1):
                        status = "✅ Safe" if c.safe_to_execute else "⚠️ Review"
                        cand_rows.append(
                            f"<tr style='border-bottom: 1px solid #f1f5f9;'>"
                            f"<td style='padding: 6px 8px;'>{idx}. <code>{c.left_column}</code> ↔ <code>{c.right_column}</code></td>"
                            f"<td style='padding: 6px 8px;'>{c.cardinality.replace('_', '-')}</td>"
                            f"<td style='padding: 6px 8px;'>{c.coverage_percentage}%</td>"
                            f"<td style='padding: 6px 8px;'><strong>{c.confidence}%</strong> ({c.confidence_level})</td>"
                            f"<td style='padding: 6px 8px;'>{status}</td>"
                            f"</tr>"
                        )
                    candidates_html = f"""
                    <div style="margin-top: 12px;">
                        <div style="font-weight: 600; font-size: 12px; color: #475569; margin-bottom: 6px;">Evaluated Candidate Keys (Ranked):</div>
                        <table style="width: 100%; border-collapse: collapse; font-size: 12px; color: #334155; text-align: left; background: #f8fafc; border-radius: 6px;">
                            <thead>
                                <tr style="border-bottom: 2px solid #e2e8f0; font-weight: 600;">
                                    <th style="padding: 6px 8px;">Key Pair</th>
                                    <th style="padding: 6px 8px;">Cardinality</th>
                                    <th style="padding: 6px 8px;">Coverage</th>
                                    <th style="padding: 6px 8px;">Confidence</th>
                                    <th style="padding: 6px 8px;">Status</th>
                                </tr>
                            </thead>
                            <tbody>
                                {"".join(cand_rows)}
                            </tbody>
                        </table>
                    </div>
                    """

                explanation_html = f"<div style='margin: 8px 0; font-size: 13px; color: #1e293b;'><strong>Analysis:</strong> {j.explanation}</div>" if j.explanation else ""

                join_card = f"""
                <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 10px; padding: 18px; margin-bottom: 14px; box-shadow: 0 1px 3px rgba(0,0,0,0.04);">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; flex-wrap: wrap; gap: 8px;">
                        <div style="font-size: 15px; font-weight: 600; color: #0f172a;">
                            🔗 {j.dataset_a} &nbsp; ↔ &nbsp; 📄 {j.dataset_b}
                        </div>
                        <div style="display: flex; gap: 8px; align-items: center;">
                            <span style="{c_style} font-size: 12px; font-weight: 700; padding: 3px 10px; border-radius: 12px;">
                                {j.confidence_level} ({j.confidence}%)
                            </span>
                            <span style="background: #f8fafc; color: #475569; border: 1px solid #cbd5e1; font-size: 12px; font-weight: 600; padding: 3px 10px; border-radius: 12px;">
                                {cardinality_label}
                            </span>
                            {card_badge}
                        </div>
                    </div>

                    <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 10px; font-size: 13px;">
                        <div style="margin-bottom: 4px; color: #334155;">
                            <strong>Candidate Join:</strong> <code>{j.dataset_a}.{j.left_column}</code> &nbsp; ↔ &nbsp; <code>{j.dataset_b}.{j.right_column}</code>
                        </div>
                        <div style="color: #64748b; font-size: 12px; display: flex; gap: 16px; flex-wrap: wrap;">
                            <span>Key Coverage: <strong>{j.coverage_percentage}%</strong></span>
                            <span>Matched Rows: <strong>{j.matched_rows:,}</strong></span>
                            <span>Unmatched ({j.dataset_b}): <strong>{j.unmatched_right_rows:,}</strong></span>
                        </div>
                    </div>

                    {explanation_html}

                    <div style="margin-top: 8px;">
                        <div style="font-weight: 600; font-size: 12px; color: #475569; margin-bottom: 4px;">Deterministic Evidence:</div>
                        <ul style="margin: 0; padding-left: 20px; font-size: 12px; color: #475569;">
                            {evidence_list}
                        </ul>
                    </div>

                    {warnings_html}
                    {candidates_html}
                </div>
                """
                join_cards.append(join_card)

            join_section_html = f"""
            <div style="margin-top: 24px;">
                <h3 style="margin: 0 0 12px 0; color: #0f172a; font-size: 18px; display: flex; align-items: center; gap: 8px;">
                    ⚡ Safe Join Intelligence & Recommendations
                </h3>
                {"".join(join_cards)}
            </div>
            """

        return f"""
        <div style="margin: 16px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;">
            <div style="margin-bottom: 20px;">
                <h3 style="margin: 0 0 12px 0; color: #0f172a; font-size: 18px; display: flex; align-items: center; gap: 8px;">
                    📂 Multi-Dataset Workspace ({len(profiles)} Datasets)
                </h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); gap: 12px;">
                    {datasets_html}
                </div>
            </div>

            <div>
                <h3 style="margin: 0 0 12px 0; color: #0f172a; font-size: 18px; display: flex; align-items: center; gap: 8px;">
                    🔗 Dataset Compatibility & Relationships
                </h3>
                {relationships_html}
            </div>

            {join_section_html}
        </div>
        """

    @classmethod
    def format_join_execution(cls, result: Any) -> str:
        """Renders structured execution summary HTML for JoinExecutionResult."""
        if not result:
            return ""

        status_styles = {
            "SUCCESS": "background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0;",
            "SUCCESS_WITH_WARNINGS": "background: #fffbeb; color: #92400e; border: 1px solid #fde68a;",
            "BLOCKED": "background: #fef2f2; color: #991b1b; border: 1px solid #fecaca;",
        }
        status_icons = {
            "SUCCESS": "✅",
            "SUCCESS_WITH_WARNINGS": "⚠️",
            "BLOCKED": "🚫",
        }

        s_style = status_styles.get(result.status, status_styles["BLOCKED"])
        s_icon = status_icons.get(result.status, "❓")
        status_label = result.status.replace("_", " ")

        if result.status == "BLOCKED":
            return f"""
            <div style="background: #ffffff; border: 1px solid #fecaca; border-radius: 10px; padding: 18px; margin: 16px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                    <div style="font-size: 16px; font-weight: 700; color: #991b1b;">
                        {s_icon} Join Execution Blocked
                    </div>
                    <span style="{s_style} font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 14px;">
                        {status_label}
                    </span>
                </div>
                <div style="background: #fef2f2; border: 1px solid #f87171; border-radius: 8px; padding: 12px 16px; color: #7f1d1d; font-size: 13px;">
                    <strong>Reason for Block:</strong> {result.error_reason or 'Pre-join safety gate validation failed.'}
                </div>
                <div style="margin-top: 12px; font-size: 12px; color: #64748b;">
                    Join details: <code>{result.left_dataset}.{result.left_column}</code> ↔ <code>{result.right_dataset}.{result.right_column}</code> ({result.join_type})
                </div>
            </div>
            """

        evidence_items = "".join(f"<li style='margin-bottom: 2px;'>{ev}</li>" for ev in result.evidence)
        warnings_html = ""
        if result.warnings:
            warn_items = "".join(f"<li style='margin-bottom: 2px;'>{w}</li>" for w in result.warnings)
            warnings_html = f"""
            <div style="background: #fffbeb; border: 1px solid #fef08a; border-radius: 8px; padding: 10px 14px; margin-top: 12px;">
                <div style="font-weight: 600; font-size: 12px; color: #854d0e; margin-bottom: 4px;">⚠️ Post-Join Warnings:</div>
                <ul style="margin: 0; padding-left: 18px; font-size: 12px; color: #713f12;">
                    {warn_items}
                </ul>
            </div>
            """

        dup_badge = (
            "<span style='color: #dc2626; font-weight: 600;'>Detected ⚠️</span>"
            if result.duplicate_expansion_detected
            else "<span style='color: #16a34a; font-weight: 600;'>None (Safe)</span>"
        )

        return f"""
        <div style="background: #ffffff; border: 1px solid #e2e8f0; border-radius: 12px; padding: 20px; margin: 18px 0; font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; box-shadow: 0 2px 4px rgba(0,0,0,0.05);">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px; flex-wrap: wrap; gap: 8px;">
                <div style="font-size: 17px; font-weight: 700; color: #0f172a; display: flex; align-items: center; gap: 8px;">
                    {s_icon} Safe Join Executed
                </div>
                <span style="{s_style} font-size: 12px; font-weight: 700; padding: 4px 12px; border-radius: 14px;">
                    {status_label}
                </span>
            </div>

            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 12px; margin-bottom: 16px;">
                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 4px;">Left Dataset</div>
                    <div style="font-size: 14px; font-weight: 600; color: #0f172a;">📄 {result.left_dataset}</div>
                    <div style="font-size: 12px; color: #475569;">Key: <code>{result.left_column}</code></div>
                    <div style="font-size: 12px; color: #64748b;">Rows: {result.rows_before_left:,}</div>
                </div>

                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 4px;">Right Dataset</div>
                    <div style="font-size: 14px; font-weight: 600; color: #0f172a;">📄 {result.right_dataset}</div>
                    <div style="font-size: 12px; color: #475569;">Key: <code>{result.right_column}</code></div>
                    <div style="font-size: 12px; color: #64748b;">Rows: {result.rows_before_right:,}</div>
                </div>

                <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #64748b; font-weight: 600; margin-bottom: 4px;">Execution Details</div>
                    <div style="font-size: 13px; color: #1e293b;">Join: <strong>{result.join_type.replace('_', ' ')}</strong></div>
                    <div style="font-size: 12px; color: #475569;">Cardinality: <strong>{result.cardinality.replace('_', '-')}</strong></div>
                    <div style="font-size: 12px; color: #64748b;">Confidence: <strong>{result.confidence}%</strong></div>
                </div>

                <div style="background: #f0fdf4; border: 1px solid #bbf7d0; border-radius: 8px; padding: 12px;">
                    <div style="font-size: 11px; text-transform: uppercase; color: #166534; font-weight: 600; margin-bottom: 4px;">Joined Result</div>
                    <div style="font-size: 14px; font-weight: 700; color: #14532d;">{result.rows_after:,} rows · {result.columns_after} cols</div>
                    <div style="font-size: 12px; color: #166534;">Multiplication: <strong>{result.row_multiplication_factor}x</strong></div>
                    <div style="font-size: 12px; color: #166534;">Duplicates: {dup_badge}</div>
                </div>
            </div>

            <div style="background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 10px 14px; margin-bottom: 12px; font-size: 12px; color: #334155; display: flex; gap: 20px; flex-wrap: wrap;">
                <span>Matched Rows: <strong>{result.matched_rows:,}</strong></span>
                <span>Unmatched Left: <strong>{result.unmatched_left_rows:,}</strong></span>
                <span>Unmatched Right: <strong>{result.unmatched_right_rows:,}</strong></span>
            </div>

            <div style="margin-top: 8px;">
                <div style="font-weight: 600; font-size: 12px; color: #475569; margin-bottom: 4px;">Execution & Validation Evidence:</div>
                <ul style="margin: 0; padding-left: 20px; font-size: 12px; color: #475569;">
                    {evidence_items}
                </ul>
            </div>

            {warnings_html}
        </div>
        """

