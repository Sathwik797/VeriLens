class DashboardFormatter:

    @staticmethod
    def profile_card(profile):

        return f"""
<div style="display:grid;
            grid-template-columns:repeat(2,1fr);
            gap:16px;
            margin-top:10px;">

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