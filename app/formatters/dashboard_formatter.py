class DashboardFormatter:

    @staticmethod
    def profile_card(profile, sheet_name=None):
        sheet_banner = ""
        if sheet_name:
            sheet_banner = f"""
    <div style="grid-column: 1 / -1; background: #eef2ff; color: #3730a3; padding: 10px; border-radius: 8px; text-align: center; font-size: 14px; border: 1px solid #c7d2fe;">
        📊 <strong>Active Sheet:</strong> {sheet_name}
    </div>
"""

        return f"""
<div style="display:grid;
            grid-template-columns:repeat(2,1fr);
            gap:16px;
            margin-top:10px;">
{sheet_banner}
    <div style="
        background:#f5f5f5;
        padding:18px;
        border-radius:12px;
        text-align:center;
        box-shadow:0 2px 6px rgba(0,0,0,.08);">

        <h4>Rows</h4>
        <h2>{profile.rows}</h2>

    </div>

    <div style="
        background:#f5f5f5;
        padding:18px;
        border-radius:12px;
        text-align:center;
        box-shadow:0 2px 6px rgba(0,0,0,.08);">

        <h4>Columns</h4>
        <h2>{profile.columns}</h2>

    </div>

    <div style="
        background:#f5f5f5;
        padding:18px;
        border-radius:12px;
        text-align:center;
        box-shadow:0 2px 6px rgba(0,0,0,.08);">

        <h4>Duplicates</h4>
        <h2>{profile.duplicate_rows}</h2>

    </div>

    <div style="
        background:#f5f5f5;
        padding:18px;
        border-radius:12px;
        text-align:center;
        box-shadow:0 2px 6px rgba(0,0,0,.08);">

        <h4>Memory</h4>
        <h2>{profile.memory_usage_mb} MB</h2>

    </div>

</div>
"""