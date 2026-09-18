# FlowMind: Pre-Route Timing Prediction via GraphSAGE on Post-Synthesis Netlists Using Open-Source EDA Tools

> **Research portfolio** · Pre-route timing prediction with Graph Neural Networks · ISPD 2027 (in preparation)

FlowMind is a full-stack EDA flow automation framework built on open-source tools (Yosys, OpenSTA, OpenROAD, Sky130 PDK) that progressively layers machine learning onto a production-style RTL-to-signoff pipeline. It is structured as a sequence of extensions, each standing alone as a demonstrable result while collectively building toward a publishable GNN-based timing predictor.

The central research question: **can a graph neural network trained on post-synthesis netlists predict worst-case timing slack before routing and generalize to designs it has never seen?**

---

## Results at a Glance

| Metric | Value |
|---|---|
| Designs synthesized | PicoRV32, AES-128, GCD, UART, SPI, AES-ORFS |
| P&R closure (GCD) | WNS = 0 ns, 0 DRC violations |
| MCMM corners | SS / TT / FF across 3 designs |
| ML log triage | 5,749 chunks · 36 files · sub-second query |
| XGBoost slack predictor | CV MAE = 0.56 ns · R² = 0.93 |
| GNN test MAE (unseen design) | **0.96 ns** vs XGBoost **2.90 ns** |
| GNN architecture | GraphSAGE · 3 layers · 64-dim · 25k params |
| Training graphs | 19 (4 designs × clock-period variants) |
| Test design | PicoRV32 (held out from all ML training) |

---

## Repository Structure

```
flowmind/
├── run_synthesis.py          # Extension 1: RTL → synthesis → STA
├── run_pnr.py                # Extension 2: synthesis → P&R (OpenROAD)
├── run_signoff.py            # Extension 2b: MCMM STA signoff
├── run_qual.py               # Extension 3: tool qualification suite
├── run_log_triage.py         # Extension 4a: TF-IDF log search engine
├── run_slack_predictor.py    # Extension 4b: XGBoost WNS predictor
├── run_gnn_prep.py           # Extension 5: netlist → PyG graphs
├── run_gnn_train.py          # Extension 5: GraphSAGE training
├── run_gnn_eval.py           # Extension 5: GNN vs XGBoost comparison
│
├── configs/                  # Per-design YAML (clock, I/O delays, PDK paths)
├── templates/                # Jinja2: synthesize.ys, constraints.sdc, run_sta.tcl
├── scripts/                  # parse_qor.py, qor_database.py, gen_dashboard.py
│
├── ext5-gnn/
│   ├── netlists/             # Flat Yosys JSON + Sky130-mapped GL Verilog
│   └── graphs/               # Serialized PyG Data objects (.pt)
│
├── results/
│   └── qor_runs.db           # SQLite: 35 runs across 6 designs
├── outputs/                  # Per-design STA reports (.rpt), DRC reports
├── models/
│   ├── slack_predictor.pkl   # XGBoost model
│   └── gnn_slack.pt          # GraphSAGE checkpoint
└── reports/
    ├── triage_report.json
    ├── predictor_report.html
    ├── gnn_metrics.json
    └── gnn_report.html
```

---

## Tool Stack

| Tool | Version | Role | Commercial Equivalent |
|---|---|---|---|
| Yosys | 0.52 | RTL synthesis, netlist export | Cadence Genus |
| OpenSTA | 2.0.17 | Static timing analysis | Cadence Tempus / Synopsys PT |
| OpenROAD | latest | Place & route | Cadence Innovus |
| SkyWater 130nm | sky130hd | Standard cell library | TSMC / Samsung PDK |
| PyTorch Geometric | latest | GNN training | — |
| XGBoost | latest | Baseline predictor | — |
| ChromaDB / TF-IDF | — | Log triage | — |

---

## Extensions

### Extension 1 — Synthesis Flow Automation

Single-command RTL-to-STA flow with Jinja2-templated scripts, per-design YAML configs, and automatic QoR parsing into SQLite.

```bash
python run_synthesis.py --design picorv32 --period 13.0
python run_synthesis.py --design aes --period 29.0
python run_synthesis.py --design gcd
```

**What it does:**
- Renders `synthesize.ys.j2` → runs Yosys synthesis
- Renders `constraints.sdc.j2` + `run_sta.tcl.j2` → runs OpenSTA
- Parses WNS / TNS / area / cell count from logs
- Writes results to `results/qor_runs.db`
- Generates interactive Plotly dashboard at `reports/dashboard.html`

**Results:**

| Design | Cells | Area (µm²) | WNS (ns) | Max Freq |
|---|---|---|---|---|
| PicoRV32 | 8,041 | 77,250 | 0.00 | 76 MHz |
| AES-128 | 997 | 18,401 | 0.00 | 36 MHz |
| GCD | 27 | 68 | 0.00 | 212 MHz |

---

### Extension 2 — Place & Route + MCMM Signoff

Full RTL-to-routed-layout flow using OpenROAD via Docker wrapper. Multi-corner multi-mode STA signoff across SS / TT / FF corners with OCV derating.

```bash
python run_pnr.py --design gcd
python run_signoff.py --design gcd
```

**GCD P&R results:** WNS = 0 ns · Area = 2,431 µm² · Utilization = 42% · **0 DRC violations**

---

### Extension 3 — Tool Qualification Suite

Automated regression suite that validates flow reproducibility — a requirement in any production CAD environment before tool adoption.

```bash
python run_qual.py
```

Runs 6 checks across 2 designs × 2 effort levels (medium / high): WNS consistency, area delta bounds, cell count stability, runtime bounds. **6/6 checks passed.**

---

### Extension 4a — ML-Assisted Log Triage

EDA runs generate thousands of log lines across synthesis, P&R, signoff, and qualification. Finding the root cause of a timing violation or DRC error requires manual grep across 30+ files.

FlowMind replaces this with a TF-IDF search engine over all EDA logs:

```bash
python run_log_triage.py --query "WNS violation timing slack" --top 5
python run_log_triage.py --query "DRC error metal spacing" --top 5
python run_log_triage.py --reindex   # rebuild index after new runs
```

**How it works:**
- Scans 36 log / report files, splits into 20-line chunks (5,749 total)
- Strips noise: absolute paths, hex addresses, ISO timestamps
- Builds TF-IDF matrix (5,000 features, sublinear TF, scikit-learn)
- Cosine similarity search via numpy — sub-second on full corpus
- Outputs ranked JSON report to `reports/triage_report.json`

*Note: ChromaDB with neural embeddings was evaluated but rejected — EDA logs are keyword-heavy, not semantic. TF-IDF outperforms neural retrieval on this domain and runs 100× faster.*

---

### Extension 4b — XGBoost WNS Slack Predictor

Can synthesis QoR metrics predict post-STA worst-case slack? This replaces the need to run a full STA (minutes to hours) with a fast model trained on existing run data.

```bash
python run_slack_predictor.py
```

**Features:** cell count, chip area, sequential %, clock period, design identity (encoded), area-per-cell

**Training:** 22 unique (design, clock) combinations · 550 synthetic augmentations (±5% Gaussian noise on numeric features) · PicoRV32 excluded (held-out test design)

**Results:** CV MAE = 0.56 ± 0.10 ns · CV R² = 0.93 ± 0.05

**Top features by importance:**
| Feature | Importance |
|---|---|
| cell_count | 29% |
| area_per_cell | 27% |
| chip_area_um2 | 19% |
| clock_period_ns | 12% |

---

### Extension 5 — GNN Pre-Route Timing Prediction

The core research contribution. Can a graph neural network trained on post-synthesis netlists predict WNS **before routing** — and generalize to designs it has never seen?

#### Graph Construction

Each synthesized netlist is parsed from flat Yosys JSON into a directed PyG graph:

| Element | Maps to |
|---|---|
| Node | Cell instance (AND2, DFF, MUX, …) |
| Edge | Net connection: output pin → input pin |
| Node features | [cell_type_enc, is_sequential, fanout_norm, fanin_norm, pin_count_norm] |
| Label | Graph-level WNS (ns) from OpenSTA TT corner |

```bash
python run_gnn_prep.py    # builds graphs for all designs
```

#### Model Architecture

```
NetlistGraph (nodes=N, edges=E, features=5)
    ↓
Node embedding (Linear → 64-dim)
    ↓
GraphSAGE Layer 1 (64 → 64, mean aggregation, ReLU, dropout=0.1)
    ↓
GraphSAGE Layer 2 (64 → 64)
    ↓
GraphSAGE Layer 3 (64 → 64)
    ↓
Global readout: concat(mean_pool, max_pool) → 128-dim
    ↓
MLP (128 → 64 → 1)  →  predicted WNS (ns)
```

25,537 parameters total. GraphSAGE chosen for its inductive capability — it generalizes to unseen graph structures at inference time.

```bash
python run_gnn_train.py   # trains on 17 graphs, tests on PicoRV32
python run_gnn_eval.py    # GNN vs XGBoost comparison report
```

#### Dataset

| Design | Nodes | Edges | WNS (ns) | Split |
|---|---|---|---|---|
| GCD (20 ns) | 303 | 551 | +13.59 | train |
| GCD (5 ns) | 303 | 551 | −0.90 | train |
| GCD (8 ns) | 303 | 551 | +0.45 | train |
| UART (5 ns) | 562 | 1,153 | −0.14 | train |
| UART (4 ns) | 562 | 1,153 | −1.14 | train |
| UART (4.5 ns) | 562 | 1,153 | −0.64 | train |
| AES-ORFS (10 ns) | 10,469 | 27,184 | −0.60 | train |
| AES-ORFS (9 ns) | 10,469 | 27,184 | −1.60 | train |
| AES-ORFS (10.5 ns) | 10,469 | 27,184 | −0.10 | train |
| SPI (1.5 ns) | 58 | 109 | −0.11 | train |
| SPI (1.2 ns) | 58 | 109 | −0.41 | train |
| AES (31 ns) | 24,689 | 49,465 | −0.03 | train |
| … (7 more variants) | — | — | — | train |
| **PicoRV32 (13 ns)** | **8,038** | **18,360** | **−0.16** | **TEST** |

19 graphs total. PicoRV32 is held out from all ML training — neither GNN nor XGBoost sees it during training.

#### Results

| Model | Mean Train MAE | **Test MAE (PicoRV32)** |
|---|---|---|
| XGBoost (feature-based) | 1.18 ns | 2.90 ns |
| **GraphSAGE GNN** | **1.71 ns** | **0.96 ns** |

**The GNN generalizes 3× better to an unseen design.** XGBoost relies on design identity as a feature — it fails on PicoRV32 because it has never seen a 8,038-node sequential processor at training time. The GNN learns structural graph features (fanout distributions, sequential depth, connectivity patterns) that transfer across design boundaries.

#### Planned Extensions

- **Subgraph sampling** — randomly sample connected subgraphs (100–500 nodes) from large designs (PicoRV32, AES) to generate 50–100 additional structurally diverse training graphs. Each subgraph receives a pseudo-WNS label from OpenSTA arrival times. This is the primary planned dataset expansion for the ISPD 2027 submission.
- **Liberty corner variation** — SS / FF corner graphs for additional label diversity
- **Edge features** — estimated Manhattan wire length from analytical placement

---

## Quick Start

```bash
# Clone and set up environment
git clone https://github.com/das-subrata/flowmind
cd flowmind
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run synthesis flow
python run_synthesis.py --design gcd

# Run ML log triage
python run_log_triage.py --reindex --query "timing violation slack"

# Train GNN
python run_gnn_prep.py
python run_gnn_train.py
python run_gnn_eval.py
```

**Prerequisites:** Yosys 0.52, OpenSTA 2.0.17, OpenROAD (Docker), SkyWater 130nm PDK

---

## Environment

Developed and tested on:
- OS: Ubuntu 22.04 (WSL2 on Windows 11)
- Python: 3.14, venv
- PyTorch: 2.14 (CPU)
- PyTorch Geometric: latest

---

## Commit History

```
04ab34d Extension 5: fair eval - GNN 0.96ns vs XGBoost 2.90ns test MAE
b59cd2d Extension 5: GNN pre-route timing prediction - GraphSAGE baseline
9947385 Extension 4: ML log triage (TF-IDF) + XGBoost slack predictor
c268119 Extension 3: Tool qualification suite - 6/6 checks passed
e4be568 Extension 2: MCMM STA signoff across SS/TT/FF corners
9f42127 Extension 1: OpenROAD P&R flow - GCD design, RTL to routed layout
```

---

## Citation

If you use FlowMind in your research:

```bibtex
@misc{flowmind2026,
  title   = {FlowMind: ML-Augmented EDA Flow Automation with GNN-Based Pre-Route Timing Prediction},
  author  = {Das, Subrata},
  year    = {2026},
  url     = {https://github.com/das-subrata/flowmind}
}
```

---

*Built with Yosys · OpenSTA · OpenROAD · SkyWater 130nm · PyTorch Geometric*
