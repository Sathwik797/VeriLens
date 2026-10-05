class DashboardFormatter:

    @staticmethod
    def profile_card(profile):

        return f"""
        <div style="display:grid;
                    grid-template-columns:repeat(2,1fr);
                    gap:15px;">

            <div style="padding:15px;
                        border-radius:10px;
                        background:#F5F5F5;">
                <h3>Rows</h3>
                <h2>{profile.rows}</h2>
            </div>

            <div style="padding:15px;
                        border-radius:10px;
                        background:#F5F5F5;">
                <h3>Columns</h3>
                <h2>{profile.columns}</h2>
            </div>

            <div style="padding:15px;
                        border-radius:10px;
                        background:#F5F5F5;">
                <h3>Duplicates</h3>
                <h2>{profile.duplicate_rows}</h2>
            </div>

            <div style="padding:15px;
                        border-radius:10px;
                        background:#F5F5F5;">
                <h3>Memory</h3>
                <h2>{profile.memory_usage_mb} MB</h2>
            </div>

        </div>
        """