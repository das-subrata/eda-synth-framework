#!/usr/bin/env python3
"""
Extension 5: GNN Pre-Route Timing Prediction
Step 1: Parse flat Yosys JSON netlists → PyG graph objects with WNS labels.

Graph schema:
  Nodes  : one per cell instance
  Edges  : net connections (output pin → input pin), directed
  Node features : [cell_type_enc, is_sequential, fanout, fanin, pin_count]
  Label  : graph-level WNS (ns) from OpenSTA TT corner report
"""
import json, re, pickle
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
from torch_geometric.data import Data

ROOT      = Path(__file__).parent
NETLIST_D = ROOT / "ext5-gnn/netlists"
OUT_DIR   = ROOT / "ext5-gnn/graphs"; OUT_DIR.mkdir(parents=True, exist_ok=True)

DESIGNS = {
    "gcd": {
        "netlist": NETLIST_D / "gcd_flat.json",
        "sta_rpt": ROOT / "outputs/gcd/timing_TT_025C_1v80.rpt",
        "top":     "gcd",
    },
    "picorv32": {
        "netlist": NETLIST_D / "picorv32_flat.json",
        "sta_rpt": ROOT / "outputs/picorv32/timing_TT_025C_1v80.rpt",
        "top":     "picorv32",
    },
    "aes": {
        "netlist": NETLIST_D / "aes_flat.json",
        "sta_rpt": ROOT / "outputs/aes/timing_TT_025C_1v80.rpt"
                   if (ROOT / "outputs/aes/timing_TT_025C_1v80.rpt").exists()
                   else None,
        "top":     "aes",
    },
}

SEQ_TYPES = {
    "$_DFF_P_", "$_DFF_N_", "$_DFFE_PP_", "$_DFFE_PN_",
    "$_DFFE_NP_", "$_DFFE_NN_", "$_DLATCH_P_", "$_DLATCH_N_",
}

def build_cell_vocab(all_mods: list) -> dict:
    types = set()
    for mod in all_mods:
        for cell in mod["cells"].values():
            types.add(cell["type"])
    vocab = {t: i for i, t in enumerate(sorted(types))}
    print(f"[prep] Cell type vocabulary: {len(vocab)} types")
    return vocab

def parse_sta_wns(rpt_path) -> float:
    """Extract WNS from OpenSTA path report — minimum slack across all paths."""
    if rpt_path is None or not rpt_path.exists():
        print(f"[prep] WARNING: STA report not found: {rpt_path}")
        return 0.0
    text = rpt_path.read_text(errors="replace")
    slacks = re.findall(r"^\s+([-\d.]+)\s+slack\s+\(", text, re.MULTILINE)
    if not slacks:
        return 0.0
    return min(float(s) for s in slacks)

def netlist_to_pyg(design: str, cfg: dict, vocab: dict) -> Data:
    with open(cfg["netlist"]) as f:
        d = json.load(f)

    mods = d["modules"]
    top  = cfg["top"]
    if top not in mods:
        top = next((k for k in mods if k.lower() == top.lower()), list(mods.keys())[-1])
    mod   = mods[top]
    cells = mod["cells"]

    cell_names = list(cells.keys())
    cell_idx   = {n: i for i, n in enumerate(cell_names)}
    N          = len(cell_names)

    net_drivers = defaultdict(list)
    net_sinks   = defaultdict(list)

    cell_types = []
    is_seq     = []
    pin_counts = []

    for name, cell in cells.items():
        idx   = cell_idx[name]
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

    fanout_arr = np.array([fanout[i] for i in range(N)], dtype=np.float32)
    fanin_arr  = np.array([fanin[i]  for i in range(N)], dtype=np.float32)

    x = np.stack([
        np.array(cell_types,  dtype=np.float32) / max(len(vocab), 1),
        np.array(is_seq,      dtype=np.float32),
        fanout_arr / (fanout_arr.max() + 1e-6),
        fanin_arr  / (fanin_arr.max()  + 1e-6),
        np.array(pin_counts,  dtype=np.float32) / 10.0,
    ], axis=1)

    wns = parse_sta_wns(cfg["sta_rpt"])
    print(f"[prep] {design}: {N} cells, {len(src_list)} edges, WNS={wns:.4f} ns")

    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    data = Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = edge_index,
        y          = torch.tensor([wns], dtype=torch.float),
        design     = design,
        num_nodes  = N,
    )
    return data

def main():
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

    vocab_path = OUT_DIR / "vocab.pkl"
    with open(vocab_path, "wb") as f:
        pickle.dump(vocab, f)
    print(f"\n[prep] Vocab saved → {vocab_path}")

    print("\n── Graph Summary ─────────────────────────────────────────")
    for design, g in graphs.items():
        print(f"  {design:12s}  nodes={g.num_nodes:6d}  "
              f"edges={g.edge_index.shape[1]:7d}  "
              f"WNS={g.y.item():.4f} ns")

if __name__ == "__main__":
    main()
