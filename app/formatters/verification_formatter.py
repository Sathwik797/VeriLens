from typing import Any, Dict, List, Optional
from app.models.trust_score import TrustScore
from app.models.verification import AnalyticalClaim, VerificationResult


class VerificationFormatter:
    """
    Formats TrustScore, claims, and verification results into clean,
    accessible HTML for the Gradio user interface.
    """

    @staticmethod
    def trust_score_card(trust_score: TrustScore) -> str:
        """Renders the TrustScore card with status badge, bar, and explanation."""
        status = trust_score.status
        score = trust_score.score

        if status == "High trust":
            badge_bg = "#e6f4ea"
            badge_color = "#137333"
            bar_color = "#34a853"
        elif status == "Moderate trust":
            badge_bg = "#e8f0fe"
            badge_color = "#1a73e8"
            bar_color = "#4285f4"
        elif status == "Low trust":
            badge_bg = "#fef7e0"
            badge_color = "#b06000"
            bar_color = "#fbbc04"
        else:
            badge_bg = "#fce8e6"
            badge_color = "#c5221f"
            bar_color = "#ea4335"

        return f"""
<div style="background: #ffffff; border: 1px solid #e0e0e0; border-radius: 12px; padding: 18px; margin: 12px 0; box-shadow: 0 2px 6px rgba(0,0,0,0.06);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
        <span style="font-size: 1.1em; font-weight: 600; color: #333333;">🛡️ Trust Score</span>
        <span style="background: {badge_bg}; color: {badge_color}; padding: 5px 14px; border-radius: 16px; font-weight: bold; font-size: 0.95em;">
            {status}
        </span>
    </div>
    <div style="display: flex; align-items: baseline; gap: 8px; margin-bottom: 8px;">
        <span style="font-size: 2.2em; font-weight: 800; color: #1a1a1a;">{score:g}</span>
        <span style="font-size: 1.2em; color: #666666; font-weight: 500;">/ 100</span>
    </div>
    <div style="background: #eceff1; border-radius: 6px; height: 10px; overflow: hidden; margin-bottom: 10px;">
        <div style="background: {bar_color}; width: {min(max(score, 0), 100)}%; height: 100%; border-radius: 6px;"></div>
    </div>
    <p style="margin: 0; color: #555555; font-size: 0.95em; line-height: 1.4;">
        {trust_score.explanation}
    </p>
</div>
"""

    @staticmethod
    def claims_breakdown(
        claims: List[AnalyticalClaim],
        verification_results: List[VerificationResult]
    ) -> str:
        """Renders individual claim cards with verification badges and evidence details."""
        if not claims:
            return """
<div style="padding: 14px; background: #f9f9f9; border-radius: 8px; color: #666666; text-align: center; margin: 10px 0;">
    No analytical claims were identified for verification.
</div>
"""

        cards_html = []
        for i, (claim, vr) in enumerate(zip(claims, verification_results), start=1):
            if vr.status == "verified":
                status_badge = '<span style="background: #e6f4ea; color: #137333; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 0.9em;">✅ Verified</span>'
                border_accent = "#34a853"
            elif vr.status == "mismatch":
                status_badge = '<span style="background: #fce8e6; color: #c5221f; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 0.9em;">❌ Mismatch</span>'
                border_accent = "#ea4335"
            else:
                status_badge = '<span style="background: #fef7e0; color: #b06000; padding: 4px 12px; border-radius: 12px; font-weight: bold; font-size: 0.9em;">⚠️ Inconclusive</span>'
                border_accent = "#fbbc04"

            subject_text = claim.subject or "Dataset"
            comparison_text = (claim.comparison or "equal").replace("_", " ").title()
            claimed_val = f"{claim.value:g}" if isinstance(claim.value, (int, float)) else str(claim.value or "N/A")
            actual_val = f"{vr.actual_value:g}" if isinstance(vr.actual_value, (int, float)) else str(vr.actual_value or "N/A")

            card = f"""
<div style="background: #ffffff; border-left: 4px solid {border_accent}; border-top: 1px solid #e0e0e0; border-right: 1px solid #e0e0e0; border-bottom: 1px solid #e0e0e0; border-radius: 8px; padding: 14px 16px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
        <span style="font-weight: 700; color: #202124; font-size: 1.05em;">Claim #{i}: {subject_text} ({claim.metric})</span>
        {status_badge}
    </div>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; font-size: 0.9em; margin-bottom: 8px; color: #3c4043;">
        <div><strong>Comparison:</strong> {comparison_text}</div>
        <div><strong>Claimed:</strong> {claimed_val}</div>
        <div><strong>Actual Evidence:</strong> {actual_val}</div>
    </div>
    <div style="font-size: 0.88em; color: #5f6368; background: #f8f9fa; padding: 8px 12px; border-radius: 6px;">
        <strong>Verification Note:</strong> {vr.reason or 'No additional details.'}
    </div>
</div>
"""
            cards_html.append(card)

        return "".join(cards_html)
