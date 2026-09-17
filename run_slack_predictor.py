#!/usr/bin/env python3
"""
Extension 4b: XGBoost WNS Slack Predictor
Trains on QoR database runs (with synthetic augmentation for small datasets).
Usage: python run_slack_predictor.py
"""
import json, pickle, warnings
from pathlib import Path

import numpy as np
import pandas as pd
import sqlite3
from sklearn.model_selection import cross_val_score, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import xgboost as xgb
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings("ignore")

DB_PATH    = Path("results/qor_runs.db")
MODEL_DIR  = Path("models");  MODEL_DIR.mkdir(exist_ok=True)
REPORT_DIR = Path("reports"); REPORT_DIR.mkdir(exist_ok=True)
RANDOM_SEED = 42

# ── 1. Load & deduplicate ────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql("SELECT * FROM runs ORDER BY id", conn)
    conn.close()

    # Keep unique (design, clock_period) combinations
    df = df.drop_duplicates(subset=["design_name", "clock_period_ns"]).copy()
    print(f"[predictor] {len(df)} unique runs after dedup:")
    print(df[["design_name","clock_period_ns","cell_count","chip_area_um2",
              "wns_ns","max_freq_mhz"]].to_string(index=False))
    return df

# ── 2. Synthetic augmentation ─────────────────────────────────────────────────

# Replace the entire augment() function
def augment(df: pd.DataFrame, n_per_row: int = 25, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Perturb numeric features with Gaussian noise (±5%) to expand the dataset.
    critical_path_ns is constant per design (synthesis output) — not perturbed.
    WNS is derived as clock_period - critical_path to stay physically consistent.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _, r in df.iterrows():
        for _ in range(n_per_row):
            noise = lambda x, pct=0.05: x * (1 + rng.normal(0, pct))
            cp_ns  = r["critical_path_ns"]          # constant per design
            clk_ns = max(0.5, noise(r["clock_period_ns"]))
            wns    = round(clk_ns - cp_ns, 4)
            rows.append({
                "design_name":      r["design_name"],
                "clock_period_ns":  round(clk_ns, 3),
                "cell_count":       max(1, int(noise(r["cell_count"]))),
                "chip_area_um2":    round(max(1, noise(r["chip_area_um2"])), 2),
                "seq_pct":          float(np.clip(noise(r["seq_pct"]), 0, 100)),
                "wns_ns":           wns,
                "is_synthetic":     True,
            })
    aug = pd.DataFrame(rows)

    orig = df.copy()
    orig["is_synthetic"] = False

    combined = pd.concat([orig, aug], ignore_index=True)
    print(f"[predictor] Dataset after augmentation: {len(combined)} rows "
          f"({len(orig)} real + {len(aug)} synthetic)")
    return combined

# ── 3. Feature engineering ───────────────────────────────────────────────────
FEATURES = [
    "cell_count", "chip_area_um2", "seq_pct",
    "clock_period_ns", "design_enc",
    "area_per_cell",
]
TARGET = "wns_ns"

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    le = LabelEncoder()
    df = df.copy()
    df["design_enc"]   = le.fit_transform(df["design_name"])
    df["area_per_cell"] = df["chip_area_um2"] / df["cell_count"].clip(lower=1)
    return df, le

# ── 4. Train XGBoost ─────────────────────────────────────────────────────────
def train(df: pd.DataFrame):
    df, le = engineer(df)
    X = df[FEATURES].values
    y = df[TARGET].values

    model = xgb.XGBRegressor(
    n_estimators=200,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_SEED,
    verbosity=0,
    tree_method="hist",
    n_jobs=1,
)

    # Cross-validation on full augmented set
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    cv_mae = -cross_val_score(model, X, y, cv=kf,
                               scoring="neg_mean_absolute_error",n_jobs=1)
    cv_r2  =  cross_val_score(model, X, y, cv=kf, scoring="r2",n_jobs=1)
    print(f"[predictor] 5-fold CV  MAE={cv_mae.mean():.4f}±{cv_mae.std():.4f} ns"
          f"  R²={cv_r2.mean():.4f}±{cv_r2.std():.4f}")

    model.fit(X, y)
    y_pred = model.predict(X)
    print(f"[predictor] Train MAE={mean_absolute_error(y, y_pred):.4f} ns"
          f"  R²={r2_score(y, y_pred):.4f}")

    # Save
    pkl = MODEL_DIR / "slack_predictor.pkl"
    with open(pkl, "wb") as f:
        pickle.dump({"model": model, "label_encoder": le, "features": FEATURES}, f)
    print(f"[predictor] Model saved → {pkl}")

    return model, le, X, y, y_pred, cv_mae, cv_r2, df

# ── 5. HTML report ───────────────────────────────────────────────────────────
def make_report(model, le, X, y, y_pred, cv_mae, cv_r2, df):
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[
            "Predicted vs Actual WNS",
            "Feature Importance",
            "WNS Distribution by Design",
            "Residuals",
        ],
    )

    # 1) Predicted vs Actual
    fig.add_trace(go.Scatter(
        x=y, y=y_pred, mode="markers",
        marker=dict(color="steelblue", size=4, opacity=0.5),
        name="Runs",
    ), row=1, col=1)
    lim = [float(min(y.min(), y_pred.min()))-0.5,
           float(max(y.max(), y_pred.max()))+0.5]
    fig.add_trace(go.Scatter(
        x=lim, y=lim, mode="lines",
        line=dict(color="red", dash="dash"), name="Perfect"),
        row=1, col=1)

    # 2) Feature importance
    imp = model.feature_importances_
    order = np.argsort(imp)
    fig.add_trace(go.Bar(
        x=imp[order], y=[FEATURES[i] for i in order],
        orientation="h", marker_color="steelblue", name="Importance",
    ), row=1, col=2)

    # 3) WNS distribution by design
    for design in df["design_name"].unique():
        sub = df[df["design_name"] == design]
        fig.add_trace(go.Box(
            y=sub["wns_ns"], name=design, boxpoints="outliers",
        ), row=2, col=1)

    # 4) Residuals
    residuals = y_pred - y
    fig.add_trace(go.Scatter(
        x=y, y=residuals, mode="markers",
        marker=dict(color="orange", size=4, opacity=0.5), name="Residual",
    ), row=2, col=2)
    fig.add_hline(y=0, line_dash="dash", line_color="red", row=2, col=2)

    fig.update_layout(
        title=f"FlowMind · XGBoost Slack Predictor  |  "
              f"CV MAE={cv_mae.mean():.4f} ns  R²={cv_r2.mean():.4f}",
        height=750, showlegend=False,
        template="plotly_white",
    )

    # Metrics JSON sidebar
    metrics = {
        "cv_mae_mean_ns": round(float(cv_mae.mean()), 4),
        "cv_mae_std_ns":  round(float(cv_mae.std()),  4),
        "cv_r2_mean":     round(float(cv_r2.mean()),  4),
        "cv_r2_std":      round(float(cv_r2.std()),   4),
        "n_real_runs":    int((~df["is_synthetic"]).sum()),
        "n_synthetic":    int(df["is_synthetic"].sum()),
        "features":       FEATURES,
        "feature_importance": {
            FEATURES[i]: round(float(v), 4)
            for i, v in enumerate(model.feature_importances_)
        },
        "note": "Synthetic augmentation (±5% Gaussian noise) used to expand "
                "small dataset of real EDA runs.",
    }
    Path("reports/predictor_metrics.json").write_text(json.dumps(metrics, indent=2))

    html = f"""<!DOCTYPE html><html><head>
<title>FlowMind Slack Predictor</title>
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
</head><body style="font-family:sans-serif;padding:20px">
<h2>FlowMind · XGBoost WNS Slack Predictor</h2>
<p><b>CV MAE:</b> {cv_mae.mean():.4f} ± {cv_mae.std():.4f} ns &nbsp;|&nbsp;
   <b>R²:</b> {cv_r2.mean():.4f} ± {cv_r2.std():.4f} &nbsp;|&nbsp;
   <b>Real runs:</b> {metrics['n_real_runs']} &nbsp;|&nbsp;
   <b>Augmented:</b> {metrics['n_synthetic']}</p>
<p style="color:#666;font-size:0.9em">{metrics['note']}</p>
{fig.to_html(full_html=False, include_plotlyjs=False)}
</body></html>"""

    out = REPORT_DIR / "predictor_report.html"
    out.write_text(html)
    print(f"[predictor] Report → {out}")
    print(f"[predictor] Metrics → reports/predictor_metrics.json")
    return metrics

# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    df_raw = load_data()
    df_aug = augment(df_raw)
    model, le, X, y, y_pred, cv_mae, cv_r2, df_aug = train(df_aug)
    metrics = make_report(model, le, X, y, y_pred, cv_mae, cv_r2, df_aug)
    print("\n── Extension 4b complete ──────────────────────────────────")
    print(json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
