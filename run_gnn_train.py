#!/usr/bin/env python3
"""
Extension 5: GNN Pre-Route Timing Prediction
Step 2: Train GraphSAGE on netlist graphs, predict graph-level WNS.

Train: gcd + aes (if labeled)
Test:  picorv32  (cross-design generalization)
"""
import json, pickle
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sequential
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv, global_mean_pool, global_max_pool
from torch_geometric.utils import add_self_loops
import numpy as np

ROOT     = Path(__file__).parent
GRAPH_D  = ROOT / "ext5-gnn/graphs"
MODEL_D  = ROOT / "models"; MODEL_D.mkdir(exist_ok=True)
REPORT_D = ROOT / "reports"; REPORT_D.mkdir(exist_ok=True)

EPOCHS      = 300
LR          = 1e-3
HIDDEN      = 64
NUM_LAYERS  = 3
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

# ── Model ─────────────────────────────────────────────────────────────────────
class TimingGNN(torch.nn.Module):
    def __init__(self, in_channels: int, hidden: int = 64, num_layers: int = 3):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden))
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden, hidden))

        self.mlp = Sequential(
            Linear(hidden * 2, hidden),   # *2: mean + max pooling concat
            ReLU(),
            Linear(hidden, 1),
        )

    def forward(self, x, edge_index, batch):
        for conv in self.convs:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.1, training=self.training)

        # Graph-level readout: concat mean and max pooling
        x = torch.cat([
            global_mean_pool(x, batch),
            global_max_pool(x, batch),
        ], dim=1)
        return self.mlp(x).squeeze(-1)

# ── Data loading ──────────────────────────────────────────────────────────────
def load_graph(design: str) -> Data:
    path = GRAPH_D / f"{design}.pt"
    if not path.exists():
        raise FileNotFoundError(f"Graph not found: {path}. Run run_gnn_prep.py first.")
    return torch.load(path, weights_only=False)

def add_batch(g: Data) -> Data:
    """Add batch vector (all zeros — single graph) for pooling."""
    g.batch = torch.zeros(g.num_nodes, dtype=torch.long)
    return g

# ── Training ──────────────────────────────────────────────────────────────────
def train_epoch(model, optimizer, graphs: list[Data]) -> float:
    model.train()
    total_loss = 0.0
    for g in graphs:
        optimizer.zero_grad()
        pred = model(g.x, g.edge_index, g.batch)
        loss = F.mse_loss(pred, g.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(graphs)

@torch.no_grad()
def evaluate(model, graphs: list[Data]) -> dict:
    model.eval()
    results = {}
    for g in graphs:
        pred = model(g.x, g.edge_index, g.batch).item()
        actual = g.y.item()
        mae  = abs(pred - actual)
        results[g.design] = {
            "predicted_wns": round(pred, 4),
            "actual_wns":    round(actual, 4),
            "mae_ns":        round(mae, 4),
        }
    return results

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    # Load graphs
    gcd      = add_batch(load_graph("gcd"))
    picorv32 = add_batch(load_graph("picorv32"))
    aes      = load_graph("aes")

    # Only use aes if it has a real WNS label
    train_graphs = [gcd]
    if aes.y.item() != 0.0:
        train_graphs.append(add_batch(aes))
        print("[train] Training on: gcd, aes")
    else:
        print("[train] Training on: gcd only (aes has no STA label)")
    print(f"[train] Test design: picorv32 (cross-design generalization)")

    in_channels = gcd.x.shape[1]
    model = TimingGNN(in_channels=in_channels, hidden=HIDDEN, num_layers=NUM_LAYERS)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.5)

    print(f"[train] Model: {sum(p.numel() for p in model.parameters())} params  "
          f"in_channels={in_channels}  hidden={HIDDEN}  layers={NUM_LAYERS}")

    # Training loop
    history = []
    for epoch in range(1, EPOCHS + 1):
        loss = train_epoch(model, optimizer, train_graphs)
        scheduler.step()
        if epoch % 50 == 0 or epoch == 1:
            val = evaluate(model, train_graphs)
            test = evaluate(model, [picorv32])
            train_mae = np.mean([v["mae_ns"] for v in val.values()])
            test_mae  = test["picorv32"]["mae_ns"]
            print(f"  Epoch {epoch:3d}  loss={loss:.6f}  "
                  f"train_MAE={train_mae:.4f} ns  "
                  f"test_MAE(picorv32)={test_mae:.4f} ns")
        history.append(loss)

    # Final evaluation
    print("\n── Final Evaluation ──────────────────────────────────────")
    all_results = evaluate(model, train_graphs + [picorv32])
    for design, r in all_results.items():
        split = "TRAIN" if design != "picorv32" else "TEST "
        print(f"  [{split}] {design:12s}  "
              f"predicted={r['predicted_wns']:+.4f} ns  "
              f"actual={r['actual_wns']:+.4f} ns  "
              f"MAE={r['mae_ns']:.4f} ns")

    # Save model
    model_path = MODEL_D / "gnn_slack.pt"
    torch.save({
        "model_state": model.state_dict(),
        "in_channels": in_channels,
        "hidden":      HIDDEN,
        "num_layers":  NUM_LAYERS,
        "epoch":       EPOCHS,
    }, model_path)
    print(f"\n[train] Model saved → {model_path}")

    # Save metrics
    metrics = {
        "architecture": {
            "type":       "GraphSAGE",
            "in_channels": in_channels,
            "hidden":      HIDDEN,
            "num_layers":  NUM_LAYERS,
            "pooling":     "mean+max concat",
            "params":      sum(p.numel() for p in model.parameters()),
        },
        "training": {
            "epochs":       EPOCHS,
            "lr":           LR,
            "train_designs": [g.design for g in train_graphs],
            "test_design":  "picorv32",
        },
        "results":  all_results,
        "loss_history": [round(l, 6) for l in history],
    }
    metrics_path = REPORT_D / "gnn_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"[train] Metrics saved → {metrics_path}")

if __name__ == "__main__":
    main()
