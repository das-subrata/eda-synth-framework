"""
gen_dashboard.py
================
Generates a standalone HTML dashboard showing QoR trends
across all designs and runs stored in the database.

Usage:
    python3 scripts/gen_dashboard.py
    
Output:
    dashboard/index.html  (open in any browser)
"""
import sys
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent))
from qor_database import query_all_runs, query_design_runs

OUTPUT_DIR = Path(__file__).parent.parent / "dashboard"


def get_designs(runs: list) -> list:
    """Get unique design names from runs."""
    seen = set()
    designs = []
    for r in runs:
        if r["design_name"] not in seen:
            designs.append(r["design_name"])
            seen.add(r["design_name"])
    return designs


def make_wns_chart(runs: list, designs: list) -> go.Figure:
    """
    WNS vs Clock Period for each design.
    Shows timing closure point clearly.
    """
    fig = go.Figure()

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    for i, design in enumerate(designs):
        design_runs = [r for r in runs if r["design_name"] == design]
        design_runs.sort(key=lambda r: r["clock_period_ns"])

        periods = [r["clock_period_ns"] for r in design_runs
                   if r["wns_ns"] is not None]
        wns     = [r["wns_ns"]          for r in design_runs
                   if r["wns_ns"] is not None]
        statuses = [r["sta_status"]     for r in design_runs
                    if r["wns_ns"] is not None]

        # Color points: green = met, red = violated
        point_colors = [
            "#2ca02c" if s == "TIMING_MET" else "#d62728"
            for s in statuses
        ]

        fig.add_trace(go.Scatter(
            x=periods,
            y=wns,
            mode="lines+markers",
            name=design,
            line=dict(color=colors[i % len(colors)], width=2),
            marker=dict(
                color=point_colors,
                size=10,
                line=dict(width=1, color="white")
            ),
            hovertemplate=(
                f"<b>{design}</b><br>"
                "Period: %{x} ns<br>"
                "WNS: %{y:.3f} ns<br>"
                "<extra></extra>"
            )
        ))

    # Add zero line (timing closure boundary)
    fig.add_hline(
        y=0,
        line_dash="dash",
        line_color="gray",
        annotation_text="Timing Closure Boundary",
        annotation_position="right"
    )

    fig.update_layout(
        title="WNS vs Clock Period (Green = Met, Red = Violated)",
        xaxis_title="Clock Period (ns)",
        yaxis_title="Worst Negative Slack (ns)",
        hovermode="x unified",
        legend=dict(x=0.02, y=0.98),
        height=400
    )

    return fig


def make_area_chart(runs: list, designs: list) -> go.Figure:
    """Bar chart comparing chip area across designs."""
    fig = go.Figure()

    # Get latest successful run per design
    latest = {}
    for r in runs:
        if (r["synth_status"] == "SUCCESS"
                and r["chip_area_um2"] is not None):
            if r["design_name"] not in latest:
                latest[r["design_name"]] = r
            elif r["id"] > latest[r["design_name"]]["id"]:
                latest[r["design_name"]] = r

    design_names = list(latest.keys())
    areas        = [latest[d]["chip_area_um2"] for d in design_names]
    cell_counts  = [latest[d]["cell_count"]    for d in design_names]

    fig.add_trace(go.Bar(
        x=design_names,
        y=areas,
        name="Chip Area",
        marker_color=["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"],
        text=[f"{a:,.0f} µm²<br>{c:,} cells"
              for a, c in zip(areas, cell_counts)],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Area: %{y:,.0f} µm²<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Chip Area Comparison (Latest Run per Design)",
        xaxis_title="Design",
        yaxis_title="Chip Area (µm²)",
        height=400,
        showlegend=False
    )

    return fig


def make_freq_chart(runs: list, designs: list) -> go.Figure:
    """Bar chart showing maximum achievable frequency per design."""
    fig = go.Figure()

    # Get the run with best (highest) max_freq per design
    best = {}
    for r in runs:
        if (r["sta_status"] == "TIMING_MET"
                and r["max_freq_mhz"] is not None):
            if r["design_name"] not in best:
                best[r["design_name"]] = r
            elif r["max_freq_mhz"] > best[r["design_name"]]["max_freq_mhz"]:
                best[r["design_name"]] = r

    if not best:
        return go.Figure().update_layout(
            title="No timing-met runs yet")

    design_names = list(best.keys())
    freqs        = [best[d]["max_freq_mhz"] for d in design_names]
    periods      = [best[d]["clock_period_ns"] for d in design_names]

    fig.add_trace(go.Bar(
        x=design_names,
        y=freqs,
        marker_color="#2ca02c",
        text=[f"{f:.1f} MHz<br>@ {p:.0f}ns"
              for f, p in zip(freqs, periods)],
        textposition="outside",
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Max Freq: %{y:.1f} MHz<br>"
            "<extra></extra>"
        )
    ))

    fig.update_layout(
        title="Maximum Frequency per Design (Best Timing-Met Run)",
        xaxis_title="Design",
        yaxis_title="Maximum Frequency (MHz)",
        height=400,
        showlegend=False
    )

    return fig


def make_runs_table(runs: list) -> go.Figure:
    """Table showing all runs."""
    headers = ["ID", "Design", "Period(ns)",
               "Cells", "Area(µm²)", "WNS(ns)",
               "TNS(ns)", "MaxFreq(MHz)", "Status"]

    rows = {h: [] for h in headers}
    for r in sorted(runs, key=lambda x: x["id"], reverse=True):
        rows["ID"].append(r["id"])
        rows["Design"].append(r["design_name"])
        rows["Period(ns)"].append(r["clock_period_ns"])
        rows["Cells"].append(r["cell_count"] or "-")
        rows["Area(µm²)"].append(
            f"{r['chip_area_um2']:,.0f}" if r["chip_area_um2"] else "-")
        rows["WNS(ns)"].append(
            f"{r['wns_ns']:.3f}" if r["wns_ns"] is not None else "-")
        rows["TNS(ns)"].append(
            f"{r['tns_ns']:.3f}" if r["tns_ns"] is not None else "-")
        rows["MaxFreq(MHz)"].append(
            f"{r['max_freq_mhz']:.1f}" if r["max_freq_mhz"] else "-")
        rows["Status"].append(r["sta_status"] or "-")

    # Color status column
    status_colors = [
        "#d4edda" if s == "TIMING_MET" else
        "#f8d7da" if s == "TIMING_VIOLATED" else
        "#fff3cd"
        for s in rows["Status"]
    ]

    fig = go.Figure(data=[go.Table(
        header=dict(
            values=headers,
            fill_color="#1f77b4",
            font=dict(color="white", size=12),
            align="left"
        ),
        cells=dict(
            values=[rows[h] for h in headers],
            fill_color=[
                ["white"] * len(runs)
                if h != "Status" else status_colors
                for h in headers
            ],
            align="left",
            font=dict(size=11)
        )
    )])

    fig.update_layout(
        title="All Runs",
        height=300
    )

    return fig


def generate_dashboard():
    """Generate the complete HTML dashboard."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("  Loading runs from database...")
    runs = query_all_runs()

    if not runs:
        print("  No runs in database. Run some flows first.")
        return

    print(f"  Found {len(runs)} runs across "
          f"{len(set(r['design_name'] for r in runs))} designs")

    designs = get_designs(runs)

    print("  Building charts...")
    fig_wns   = make_wns_chart(runs, designs)
    fig_area  = make_area_chart(runs, designs)
    fig_freq  = make_freq_chart(runs, designs)
    fig_table = make_runs_table(runs)

    # Combine into one HTML page
    from plotly.io import to_html

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>EDA Synthesis QoR Dashboard</title>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont,
                         'Segoe UI', sans-serif;
            margin: 0; padding: 20px;
            background: #f5f5f5;
            color: #333;
        }}
        h1 {{
            color: #1f77b4;
            border-bottom: 2px solid #1f77b4;
            padding-bottom: 10px;
        }}
        .subtitle {{
            color: #666;
            margin-top: -10px;
            margin-bottom: 30px;
        }}
        .chart-container {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
        }}
        .stats-bar {{
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 15px;
            margin-bottom: 20px;
        }}
        .stat-card {{
            background: white;
            border-radius: 8px;
            padding: 15px;
            text-align: center;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .stat-number {{
            font-size: 2em;
            font-weight: bold;
            color: #1f77b4;
        }}
        .stat-label {{
            color: #666;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <h1>EDA Synthesis QoR Dashboard</h1>
    <p class="subtitle">
        Open-source EDA flow: Yosys + OpenSTA + SkyWater 130nm PDK
    </p>

    <div class="stats-bar">
        <div class="stat-card">
            <div class="stat-number">{len(runs)}</div>
            <div class="stat-label">Total Runs</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">
                {len(set(r['design_name'] for r in runs))}
            </div>
            <div class="stat-label">Designs</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">
                {sum(1 for r in runs if r['sta_status'] == 'TIMING_MET')}
            </div>
            <div class="stat-label">Timing Met</div>
        </div>
        <div class="stat-card">
            <div class="stat-number">
                {sum(1 for r in runs
                     if r['sta_status'] == 'TIMING_VIOLATED')}
            </div>
            <div class="stat-label">Timing Violated</div>
        </div>
    </div>

    <div class="chart-container">
        {to_html(fig_wns, full_html=False, include_plotlyjs='cdn')}
    </div>

    <div class="grid">
        <div class="chart-container">
            {to_html(fig_area, full_html=False, include_plotlyjs=False)}
        </div>
        <div class="chart-container">
            {to_html(fig_freq, full_html=False, include_plotlyjs=False)}
        </div>
    </div>

    <div class="chart-container">
        {to_html(fig_table, full_html=False, include_plotlyjs=False)}
    </div>

</body>
</html>"""

    output_path = OUTPUT_DIR / "index.html"
    output_path.write_text(html)
    print(f"\n  Dashboard saved: {output_path}")
    print(f"  Size: {output_path.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    print("\n" + "="*55)
    print("  Generating QoR Dashboard...")
    print("="*55)
    generate_dashboard()
    print("="*55 + "\n")
