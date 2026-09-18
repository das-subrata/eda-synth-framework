#!/usr/bin/env python3
"""
Extension 5b: Subgraph Sampling for Dataset Augmentation
BFS-based connected subgraph extraction from large designs.
Label: parent design WNS from OpenSTA (design-level label per subgraph).
Usage: python run_gnn_subgraph.py
"""
import json, pickle, random
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import torch
from torch_geometric.data import Data

ROOT      = Path(__file__).parent
GRAPH_D   = ROOT / "ext5-gnn/graphs"
NETLIST_D = ROOT / "ext5-gnn/netlists"
VOCAB_F   = GRAPH_D / "vocab.pkl"

RANDOM_SEED  = 42
MIN_NODES    = 100
MAX_NODES    = 500
N_SUBGRAPHS  = 20   # per source entry

SEQ_TYPES = {
    "$_DFF_P_", "$_DFF_N_", "$_DFFE_PP_", "$_DFFE_PN_",
    "$_DFFE_NP_", "$_DFFE_NN_", "$_DLATCH_P_", "$_DLATCH_N_",
}

# Each entry is one (netlist, wns_label) pair.
# Multiple entries from the same netlist give structural diversity
# with different WNS labels from different clock periods.
SOURCES = [
    # PicoRV32 — 13ns clock (WNS=-0.16ns)
    {"tag": "picorv32_13ns", "netlist": NETLIST_D/"picorv32_flat.json",
     "top": "picorv32",       "wns": -0.16},

    # AES-ORFS — three clock periods
    {"tag": "aes_orfs_9ns",  "netlist": NETLIST_D/"aes_orfs_flat.json",
     "top": "aes_cipher_top", "wns": -1.60},
    {"tag": "aes_orfs_10ns", "netlist": NETLIST_D/"aes_orfs_flat.json",
     "top": "aes_cipher_top", "wns": -0.60},
    {"tag": "aes_orfs_12ns", "netlist": NETLIST_D/"aes_orfs_flat.json",
     "top": "aes_cipher_top", "wns":  0.45},

    # AES (large design) — 31ns clock (WNS=-0.03ns)
    {"tag": "aes_31ns",      "netlist": NETLIST_D/"aes_flat.json",
     "top": "aes",            "wns": -0.03},
]

# ── Netlist parsing ───────────────────────────────────────────────────────────
def parse_netlist(netlist_path: Path, top: str):
    with open(netlist_path) as f:
        d = json.load(f)
    mods = d["modules"]
    if top not in mods:
        top = next((k for k in mods if k.lower() == top.lower()),
                   list(mods.keys())[-1])
    cells      = mods[top]["cells"]
    cell_names = list(cells.keys())
    cell_idx   = {n: i for i, n in enumerate(cell_names)}
    N          = len(cell_names)

    net_drivers = defaultdict(list)
    net_sinks   = defaultdict(list)
    cell_types  = []
    is_seq      = []
    pin_counts  = []

    for name, cell in cells.items():
        idx   = cell_idx[name]
        ctype = cell["type"]
        cell_types.append(ctype)
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

    # Undirected adjacency for BFS; directed edges for PyG
    adj       = defaultdict(set)
    edges_dir = []
    for net_id, drivers in net_drivers.items():
        for d_idx in drivers:
            for s_idx in net_sinks.get(net_id, []):
                if d_idx != s_idx:
                    adj[d_idx].add(s_idx)
                    adj[s_idx].add(d_idx)
                    edges_dir.append((d_idx, s_idx))

    return {
        "N": N, "cell_types": cell_types, "is_seq": is_seq,
        "pin_counts": pin_counts, "adj": adj, "edges_dir": edges_dir,
    }

# ── BFS subgraph ──────────────────────────────────────────────────────────────
def bfs_subgraph(adj, N, seed, min_n, max_n, rng):
    visited = {seed}
    queue   = deque([seed])
    while queue and len(visited) < max_n:
        node = queue.popleft()
        neighbors = list(adj[node])
        rng.shuffle(neighbors)
        for nb in neighbors:
            if nb not in visited:
                visited.add(nb)
                queue.append(nb)
                if len(visited) >= max_n:
                    break
    return visited if len(visited) >= min_n else None

# ── Build PyG Data ────────────────────────────────────────────────────────────
def build_pyg(netlist, node_set, wns, vocab, design_tag):
    node_list = sorted(node_set)
    local_idx = {g: l for l, g in enumerate(node_list)}
    n         = len(node_list)

    fanout = defaultdict(int)
    fanin  = defaultdict(int)
    sub_edges = []
    for (s, d) in netlist["edges_dir"]:
        if s in local_idx and d in local_idx:
            sub_edges.append((local_idx[s], local_idx[d]))
            fanout[local_idx[s]] += 1
            fanin[local_idx[d]]  += 1

    fanout_arr = np.array([fanout[i] for i in range(n)], dtype=np.float32)
    fanin_arr  = np.array([fanin[i]  for i in range(n)], dtype=np.float32)
    ct         = netlist["cell_types"]
    iseq       = netlist["is_seq"]
    pc         = netlist["pin_counts"]

    x = np.stack([
        np.array([vocab.get(ct[g], 0)  for g in node_list],
                 dtype=np.float32) / max(len(vocab), 1),
        np.array([iseq[g]              for g in node_list], dtype=np.float32),
        fanout_arr / (fanout_arr.max() + 1e-6),
        fanin_arr  / (fanin_arr.max()  + 1e-6),
        np.array([pc[g]                for g in node_list],
                 dtype=np.float32) / 10.0,
    ], axis=1)

    ei = (torch.tensor(sub_edges, dtype=torch.long).t().contiguous()
          if sub_edges else torch.zeros((2, 0), dtype=torch.long))

    return Data(
        x          = torch.tensor(x, dtype=torch.float),
        edge_index = ei,
        y          = torch.tensor([wns], dtype=torch.float),
        design     = design_tag,
        num_nodes  = n,
    )

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    with open(VOCAB_F, "rb") as f:
        vocab = pickle.load(f)

    rng         = random.Random(RANDOM_SEED)
    total_saved = 0

    # Cache parsed netlists to avoid re-parsing same file
    netlist_cache = {}

    for src in SOURCES:
        tag     = src["tag"]
        wns     = src["wns"]
        npath   = str(src["netlist"])
        top     = src["top"]

        print(f"\n[subgraph] {tag}  WNS={wns:.4f}ns")

        if npath not in netlist_cache:
            print(f"  Parsing {src['netlist'].name}...")
            netlist_cache[npath] = parse_netlist(src["netlist"], top)
        netlist = netlist_cache[npath]
        N       = netlist["N"]
        adj     = netlist["adj"]

        saved    = 0
        attempts = 0
        max_att  = N_SUBGRAPHS * 30

        while saved < N_SUBGRAPHS and attempts < max_att:
            attempts += 1
            seed     = rng.randint(0, N - 1)
            node_set = bfs_subgraph(adj, N, seed,
                                    MIN_NODES, MAX_NODES, rng)
            if node_set is None:
                continue

            data = build_pyg(netlist, node_set, wns, vocab, tag)
            out  = GRAPH_D / f"{tag}_s{saved:03d}.pt"
            torch.save(data, out)
            saved += 1
            print(f"  [{saved:2d}/{N_SUBGRAPHS}] "
                  f"nodes={data.num_nodes} "
                  f"edges={data.edge_index.shape[1]}")

        total_saved += saved
        print(f"  Saved {saved}/{N_SUBGRAPHS} subgraphs")

    print(f"\n[subgraph] Done. Total subgraphs: {total_saved}")

    # Summary
    pts = sorted(GRAPH_D.glob("*_s[0-9]*.pt"))
    wns_vals = []
    for p in pts:
        g = torch.load(p, weights_only=False)
        wns_vals.append(g.y.item())
    if wns_vals:
        print(f"[subgraph] WNS range: {min(wns_vals):.4f} to "
              f"{max(wns_vals):.4f} ns across {len(pts)} subgraphs")

if __name__ == "__main__":
    main()
