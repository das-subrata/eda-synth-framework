#!/usr/bin/env python3
"""
Extension 5: GNN Pre-Route Timing Prediction
Step 1: Parse flat Yosys JSON netlists → PyG graph objects with STA slack labels.

Graph schema:
  Nodes  : one per cell instance
  Edges  : net connections (output pin → input pin), directed
  Node features : [cell_type_enc, is_sequential, fanout, fanin, pin_count]
  Labels : per-endpoint slack from OpenSTA TT corner report
            (0.0 for non-endpoint nodes)
"""
import json, re, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data

ROOT      = Path(__file__).parent.parent
NETLIST_D = Path(__file__).parent / "netlists"
OUT_DIR   = Path(__file__).parent / "graphs"; OUT_DIR.mkdir(exist_ok=True)

DESIGNS = {
    "gcd":      {
        "netlist": NETLIST_D / "gcd_flat.json",
        "sta_rpt": ROOT / "outputs/gcd/timing_TT_025C_1v80.rpt",
        "top":     "gcd",
    },
    "picorv32": {
        "netlist": NETLIST_D / "picorv32_flat.json",
        "sta_rpt": ROOT / "outputs/picorv32/timing_TT_025C_1v80.rpt",
        "top":     "picorv32",
    },
    "aes":      {
        "netlist": NETLIST_D / "aes_flat.json",
        "sta_rpt": ROOT / "outputs/aes/timing_TT_025C_1v80.rpt" 
                   if (ROOT / "outputs/aes/timing_TT_025C_1v80.rpt").exists()
                   else None,
        "top":     "aes",
    },
}

# Primitive cell types → index (sequential cells flagged separately)
SEQ_TYPES = {"$_DFF_P_", "$_DFF_N_", "$_DFFE_PP_", "$_DFFE_PN_",
             "$_DFFE_NP_", "$_DFFE_NN_", "$_DLATCH_P_", "$_DLATCH_N_"}

def build_cell_vocab(all_netlists: list[dict]) -> dict[str, int]:
    types = set()
    for mod in all_netlists:
        for cell in mod["cells"].values():
            types.add(cell["type"])
    vocab = {t: i for i, t in enumerate(sorted(types))}
    print(f"[prep] Cell type vocabulary: {len(vocab)} types")
    return vocab

def parse_sta_slack(rpt_path: Path) -> dict[str, float]:
    """Extract {endpoint_name: slack_ns} from OpenSTA path report."""
    if rpt_path is None or not rpt_path.exists():
        return {}
    text = rpt_path.read_text(errors="replace")
    slacks = {}
    endpoint = None
    for line in text.splitlines():
        m = re.match(r"^Endpoint:\s+(\S+)", line)
        if m:
            endpoint = m.group(1)
        m = re.match(r"^\s+([-\d.]+)\s+slack\s+\(", line)
        if m and endpoint:
            slacks[endpoint] = float(m.group(1))
            endpoint = None
    return slacks

def netlist_to_pyg(design: str, cfg: dict, vocab: dict) -> Data:
    with open(cfg["netlist"]) as f:
        d = json.load(f)

    top  = cfg["top"]
    # Yosys sometimes capitalises or mangles top name; find it
    mods = d["modules"]
    if top not in mods:
        top = next((k for k in mods if k.lower() == top.lower()), list(mods.keys())[-1])
    mod  = mods[top]
    cells = mod["cells"]

    # ── Build net → {driver, sinks} map ──────────────────────────────────
    # net_id (int) → list of (cell_idx, port_name, direction)
    net_drivers = defaultdict(list)   # net_id → cell_idx
    net_sinks   = defaultdict(list)   # net_id → [cell_idx, ...]

    cell_names = list(cells.keys())
    cell_idx   = {n: i for i, n in enumerate(cell_names)}
    N          = len(cell_names)

    # Node features
    cell_types  = []
    is_seq      = []
    pin_counts  = []

    for name, cell in cells.items():
        idx  = cell_idx[name]
        ctype = cell["type"]
        cell_types.append(vocab.get(ctype, 0))
        is_seq.append(1 if ctype in SEQ_TYPES else 0)
        pin_counts.append(len(cell["connections"]))

        for port, nets in cell["connections"].items():
            direction = cell["port_directions"].get(port, "input")
            for net_id in nets:
                if isinstance(net_id, int):
                    if direction == "output":
                        net_drivers[net_id].append(idx)
                    else:
                        net_sinks[net_id].append(idx)

    # ── Build edge list ───────────────────────────────────────────────────
    src_list, dst_list = [], []
    fanout = defaultdict(int)
    fanin  = defaultdict(int)

    for net_id, drivers in net_drivers.items():
        sinks = net_sinks.get(net_id, [])
        for d_idx in drivers:
            for s_idx in sinks:
                if d_idx != s_idx:
                    src_list.append(d_idx)
                    dst_list.append(s_idx)
                    fanout[d_idx] += 1
                    fanin[s_idx]  += 1

    # ── STA slack labels ──────────────────────────────────────────────────
    sta_slacks = parse_sta_slack(cfg["sta_rpt"])
    # Map endpoint names to cell indices (best-effort substring match)
    node_slack = np.zeros(N, dtype=np.float32)
    is_endpoint = np.zeros(N, dtype=np.float32)
    matched = 0
    for ep, slack in sta_slacks.items():
        # endpoint format: "some/hier/path/_NN_" or "port_name[k]"
        # strip bit index and hierarchy, match against cell name suffix
        ep_clean = ep.split("/")[-1].split("[")[0]
        for name, idx in cell_idx.items():
            if ep_clean in name or name in ep_clean:
                node_slack[idx]   = slack
                is_endpoint[idx]  = 1.0
                matched += 1
                break
    print(f"[prep] {design}: {N} cells, {len(src_list)} edges, "
          f"{len(sta_slacks)} STA endpoints, {matched} matched to nodes")

    # ── Assemble feature matrix ───────────────────────────────────────────
    fanout_arr = np.array([fanout[i] for i in range(N)], dtype=np.float32)
    fanin_arr  = np.array([fanin[i]  for i in range(N)], dtype=np.float32)

    x = np.stack([
        np.array(cell_types,  dtype=np.float32) / max(len(vocab), 1),
        np.array(is_seq,      dtype=np.float32),
        fanout_arr / (fanout_arr.max() + 1e-6),
        fanin_arr  / (fanin_arr.max()  + 1e-6),
        np.array(pin_counts,  dtype=np.float32) / 10.0,
    ], axis=1)

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    data = Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor(node_slack,   dtype=torch.float),
        is_endpoint= torch.tensor(is_endpoint,  dtype=torch.float),
        design     = design,
        num_nodes  = N,
    )
    return data

def main():
    # Load all netlists to build shared vocab
    print("[prep] Building cell type vocabulary across all designs...")
    all_mods = []
    for cfg in DESIGNS.values():
        with open(cfg["netlist"]) as f:
            d = json.load(f)
        mods = d["modules"]
        top  = cfg["top"]
        if top not in mods:
            top = next((k for k in mods if k.lower() == top.lower()), list(mods.keys())[-1])
        all_mods.append(mods[top])
    vocab = build_cell_vocab(all_mods)

    graphs = {}
    for design, cfg in DESIGNS.items():
        print(f"\n[prep] Processing {design}...")
        g = netlist_to_pyg(design, cfg, vocab)
        graphs[design] = g
        out = OUT_DIR / f"{design}.pt"
        torch.save(g, out)
        print(f"[prep] Saved → {out}")

    # Save vocab
    vocab_path = OUT_DIR / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"\n[prep] Vocab saved → {vocab_path}")

    # Summary
    print("\n── Graph Summary ─────────────────────────────────────────")
    for design, g in graphs.items():
        ep = int(g.is_endpoint.sum())
        print(f"  {design:12s}  nodes={g.num_nodes:6d}  "
              f"edges={g.edge_index.shape[1]:7d}  "
              f"endpoints={ep:4d}  "
              f"slack_range=[{g.y.min():.2f}, {g.y.max():.2f}]")

if __name__ == "__main__":
    main()
