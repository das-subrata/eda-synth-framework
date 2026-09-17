#!/usr/bin/env python3
"""
Extension 5: GNN Pre-Route Timing Prediction
Step 3: Evaluate GNN vs XGBoost baseline, generate HTML report.
"""
import json, pickle
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT     = Path(__file__).parent
GRAPH_D  = ROOT / "ext5-gnn/graphs"
MODEL_D  = ROOT / "models"
REPORT_D = ROOT / "reports"; REPORT_D.mkdir(exist_ok=True)

# ── Reproduce model class (must match run_gnn_train.py) ──────────────────────
class TimingGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden=64, num_layers=3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))
        self.mlp = Sequential(
            Linear(hidden * 2, hidden),
            ReLU(),
            Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
        x = torch.cat([global_mean_pool(x, batch),
                        global_max_pool(x, batch)], dim=1)
        return self.mlp(x).squeeze(-1)

# ── Load GNN ──────────────────────────────────────────────────────────────────
def load_gnn():
    ckpt  = torch.load(MODEL_D / "gnn_slack.pt", weights_only=False)
    model = TimingGNN(ckpt["in_channels"], ckpt["hidden"], ckpt["num_layers"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

@torch.no_grad()
def gnn_predict(model, design: str) -> float:
    g = torch.load(GRAPH_D / f"{design}.pt", weights_only=False)
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
    return model(g.x, g.edge_index, g.batch).item()

def gnn_actual(design: str) -> float:
    g = torch.load(GRAPH_D / f"{design}.pt", weights_only=False)
    return g.y.item()

# ── Load XGBoost baseline ─────────────────────────────────────────────────────
def load_xgb():
    with open(MODEL_D / "slack_predictor.pkl", "rb") as f:
        return pickle.load(f)

def xgb_predict(bundle: dict, design: str, clock_ns: float) -> float:
    import sqlite3, pandas as pd
    conn = sqlite3.connect(ROOT / "results/qor_runs.db")
    df   = pd.read_sql("SELECT * FROM runs", conn); conn.close()
    df   = df.drop_duplicates(subset=["design_name", "clock_period_ns"])

    # Strip period suffix to get base design name (gcd_5ns → gcd)
    base = design.split("_")[0] if "_" in design else design
    # aes_orfs is a special case
    if design.startswith("aes_orfs"):
        base = "aes_orfs"

    row = df[(df["design_name"] == base) &
             (df["clock_period_ns"] == clock_ns)]
    if row.empty:
        row = df[df["design_name"] == base]
    if row.empty:
        return float("nan")
    row = row.iloc[0]

    le  = bundle["label_encoder"]
    enc = le.transform([base])[0] if base in le.classes_ else 0
    x   = np.array([[
        row["cell_count"],
        row["chip_area_um2"],
        row["seq_pct"],
        clock_ns,
        enc,
        row["chip_area_um2"] / max(row["cell_count"], 1),
    ]])
    return float(bundle["model"].predict(x)[0])

# ── Build comparison table ────────────────────────────────────────────────────
EVAL_DESIGNS = {
    "gcd":             {"clock_ns": 20.0,  "split": "train"},
    "gcd_5ns":         {"clock_ns": 5.0,   "split": "train"},
    "gcd_8ns":         {"clock_ns": 8.0,   "split": "train"},
    "uart":            {"clock_ns": 5.0,   "split": "train"},
    "uart_4ns":        {"clock_ns": 4.0,   "split": "train"},
    "uart_4p5ns":      {"clock_ns": 4.5,   "split": "train"},
    "aes_orfs":        {"clock_ns": 10.0,  "split": "train"},
    "aes_orfs_9ns":    {"clock_ns": 9.0,   "split": "train"},
    "spi":             {"clock_ns": 1.5,   "split": "train"},
    "spi_1p2ns":       {"clock_ns": 1.2,   "split": "train"},
    "picorv32":        {"clock_ns": 13.0,  "split": "test"},
    "aes":             {"clock_ns": 31.0,  "split": "train"},
}
def main():
    gnn_model  = load_gnn()
    xgb_bundle = load_xgb()

    rows = []
    for design, cfg in EVAL_DESIGNS.items():
        actual   = gnn_actual(design)
        gnn_pred = gnn_predict(gnn_model, design)
        xgb_pred = xgb_predict(xgb_bundle, design, cfg["clock_ns"])
        if actual == 0.0 and design == "aes":
            actual = None   # no ground truth for aes
        rows.append({
            "design":    design,
            "split":     cfg["split"],
            "actual":    actual,
            "gnn_pred":  round(gnn_pred, 4),
            "xgb_pred":  round(xgb_pred, 4) if not np.isnan(xgb_pred) else None,
            "gnn_mae":   round(abs(gnn_pred - actual), 4) if actual is not None else None,
            "xgb_mae":   round(abs(xgb_pred - actual), 4)
                         if (actual is not None and not np.isnan(xgb_pred)) else None,
        })

    print("\n── GNN vs XGBoost WNS Prediction ─────────────────────────")
    print(f"{'Design':12s} {'Split':6s} {'Actual':>10s} "
          f"{'GNN':>10s} {'XGB':>10s} {'GNN_MAE':>10s} {'XGB_MAE':>10s}")
    print("-" * 72)
    for r in rows:
        def fmt(v): return f"{v:+.4f}" if v is not None else "    N/A "
        print(f"{r['design']:12s} {r['split']:6s} {fmt(r['actual']):>10s} "
              f"{fmt(r['gnn_pred']):>10s} {fmt(r['xgb_pred']):>10s} "
              f"{fmt(r['gnn_mae']):>10s} {fmt(r['xgb_mae']):>10s}")

    # ── Plotly report ─────────────────────────────────────────────────────
    labeled = [r for r in rows if r["actual"] is not None]
    designs  = [r["design"] for r in labeled]
    actual   = [r["actual"]   for r in labeled]
    gnn_vals = [r["gnn_pred"] for r in labeled]
    xgb_vals = [r["xgb_pred"] for r in labeled]
    gnn_maes = [r["gnn_mae"]  for r in labeled]
    xgb_maes = [r["xgb_mae"]  for r in labeled]

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=["Predicted vs Actual WNS", "MAE Comparison"],
    )

    # Scatter: predicted vs actual
    fig.add_trace(go.Scatter(
        x=actual, y=gnn_vals, mode="markers+text",
        text=designs, textposition="top center",
        marker=dict(color="steelblue", size=12),
        name="GNN",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=actual, y=xgb_vals, mode="markers+text",
        text=designs, textposition="bottom center",
        marker=dict(color="orange", size=12, symbol="diamond"),
        name="XGBoost",
    ), row=1, col=1)
    lim = [min(actual) - 1, max(actual) + 1]
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color="red", dash="dash"), name="Perfect",
    ), row=1, col=1)

    # Bar: MAE per design
    fig.add_trace(go.Bar(
        x=designs, y=gnn_maes, name="GNN MAE",
        marker_color="steelblue",
    ), row=1, col=2)
    fig.add_trace(go.Bar(
        x=designs, y=xgb_maes, name="XGBoost MAE",
        marker_color="orange",
    ), row=1, col=2)

    fig.update_xaxes(title_text="Actual WNS (ns)", row=1, col=1)
    fig.update_yaxes(title_text="Predicted WNS (ns)", row=1, col=1)
    fig.update_yaxes(title_text="MAE (ns)", row=1, col=2)
    fig.update_layout(
        title="FlowMind · GNN vs XGBoost Pre-Route Timing Prediction",
        height=500, template="plotly_white", barmode="group",
    )

    # Save metrics
    summary = {
        "note": "GNN trained on gcd only (1 graph). XGBoost trained on augmented QoR DB.",
        "designs": rows,
        "gnn_mean_mae_ns":  round(np.mean([r["gnn_mae"] for r in labeled]), 4),
        "xgb_mean_mae_ns":  round(np.mean([r["xgb_mae"] for r in labeled
                                            if r["xgb_mae"] is not None]), 4),
    }
    with open(REPORT_D / "gnn_eval_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    html = f"""<!DOCTYPE html><html><head>
<title>FlowMind GNN Eval</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body style="font-family:sans-serif;padding:20px">
<h2>FlowMind · GNN vs XGBoost Pre-Route Timing Prediction</h2>
<p><b>GNN mean MAE:</b> {summary['gnn_mean_mae_ns']:.4f} ns &nbsp;|&nbsp;
   <b>XGBoost mean MAE:</b> {summary['xgb_mean_mae_ns']:.4f} ns</p>
<p style="color:#666;font-size:0.9em">{summary['note']}</p>
{fig.to_html(full_html=False, include_plotlyjs=False)}
<h3>Results Table</h3>
<table border="1" cellpadding="6" style="border-collapse:collapse">
<tr><th>Design</th><th>Split</th><th>Actual WNS</th>
    <th>GNN Pred</th><th>XGB Pred</th>
    <th>GNN MAE</th><th>XGB MAE</th></tr>
{''.join(f"""<tr>
  <td>{r['design']}</td><td>{r['split']}</td>
  <td>{r['actual'] if r['actual'] is not None else 'N/A'}</td>
  <td>{r['gnn_pred']}</td><td>{r['xgb_pred'] if r['xgb_pred'] else 'N/A'}</td>
  <td>{r['gnn_mae'] if r['gnn_mae'] else 'N/A'}</td>
  <td>{r['xgb_mae'] if r['xgb_mae'] else 'N/A'}</td>
</tr>""" for r in rows)}
</table>
</body></html>"""

    out = REPORT_D / "gnn_report.html"
    out.write_text(html)
    print(f"\n[eval] Report  → {out}")
    print(f"[eval] Metrics → {REPORT_D / 'gnn_eval_metrics.json'}")
    print(f"\n[eval] GNN mean MAE:      {summary['gnn_mean_mae_ns']:.4f} ns")
    print(f"[eval] XGBoost mean MAE:  {summary['xgb_mean_mae_ns']:.4f} ns")

if __name__ == "__main__":
    main()
