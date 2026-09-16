#!/usr/bin/env python3
"""
run_pnr.py
==========
Place and Route automation using OpenROAD.
Extends the FlowMind framework beyond synthesis to full P&R.

Usage:
    python3 run_pnr.py --design gcd
    python3 run_pnr.py --design gcd --utilization 0.5

Flow:
    1. Read design config from configs/<design>.yaml
    2. Run synthesis first (calls run_synthesis.py logic)
    3. Generate OpenROAD P&R script from Jinja2 template
    4. Run OpenROAD (floorplan -> place -> CTS -> route)
    5. Parse results and update QoR database
"""
import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from parse_qor import parse_yosys_log, parse_sta_log, print_qor_summary
from qor_database import insert_run

# ── Paths ──────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CONFIGS_DIR   = BASE_DIR / "configs"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUTS_DIR   = BASE_DIR / "outputs"
LOGS_DIR      = BASE_DIR / "logs"


def load_config(design_name: str,
                utilization_override: float = None) -> dict:
    config_path = CONFIGS_DIR / f"{design_name}.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if utilization_override:
        cfg["core_utilization"] = utilization_override
    # Check P&R required fields
    for field in ["tech_lef", "cell_lef", "core_utilization"]:
        if field not in cfg:
            print(f"ERROR: '{field}' missing from {config_path}")
            print("  Add tech_lef, cell_lef, core_utilization to config")
            sys.exit(1)
    return cfg


def render_template(template_name: str, context: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template(template_name)
    return template.render(**context)


def run_tool(command: str, log_path: Path, tool_name: str) -> bool:
    print(f"\n  Running {tool_name}...")
    print(f"  Log: {log_path}")
    print("  " + "-"*50)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            command, shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True)
        for line in process.stdout:
            print("  " + line, end="")
            log_file.write(line)
        process.wait()
    if process.returncode != 0:
        print(f"\n  ERROR: {tool_name} failed (exit {process.returncode})")
        return False
    print(f"\n  {tool_name} completed successfully.")
    return True


def parse_pnr_log(log_path: Path) -> dict:
    """Parse OpenROAD P&R log for key metrics."""
    metrics = {
        "wns_ns":        None,
        "tns_ns":        None,
        "design_area":   None,
        "drc_violations": None,
        "status":        "UNKNOWN"
    }
    try:
        content = log_path.read_text()

        import re

        # WNS: "wns -1.23" or "wns 0.00"
        m = re.search(r"^wns\s+([-\d.]+)", content, re.MULTILINE)
        if m:
            metrics["wns_ns"] = float(m.group(1))

        # TNS
        m = re.search(r"^tns\s+([-\d.]+)", content, re.MULTILINE)
        if m:
            metrics["tns_ns"] = float(m.group(1))

        # Design area: "Design area X um^2 Y% utilization"
        m = re.search(r"Design area\s+([\d.]+)\s+u", content)
        if m:
            metrics["design_area"] = float(m.group(1))

        # DRC violations
        m = re.search(r"(\d+)\s+violations", content)
        if m:
            metrics["drc_violations"] = int(m.group(1))

        # Status
        if "P&R complete" in content:
            metrics["status"] = "SUCCESS"
            if metrics["wns_ns"] is not None:
                metrics["status"] = (
                    "TIMING_MET" if metrics["wns_ns"] >= 0
                    else "TIMING_VIOLATED")
        elif "ERROR" in content or "error" in content.lower():
            metrics["status"] = "FAILED"

    except FileNotFoundError:
        metrics["status"] = "LOG_NOT_FOUND"

    return metrics


def run_synthesis_for_pnr(design_name: str, cfg: dict,
                           design_out: Path,
                           design_logs: Path) -> bool:
    """Run synthesis to get netlist for P&R."""
    print("\n[1/3] Running synthesis (prerequisite for P&R)...")

    # Generate synthesis script
    synth_context = {
        **cfg,
        "timestamp":      datetime.now().isoformat(),
        "output_netlist": str(design_out / f"{design_name}_synth.v"),
    }
    synth_script = design_out / "synthesize.ys"
    synth_script.write_text(
        render_template("synthesize.ys.j2", synth_context))

    # Generate SDC
    sdc_path = design_out / f"{design_name}.sdc"
    sdc_path.write_text(
        render_template("constraints.sdc.j2", synth_context))

    synth_ok = run_tool(
        f"yosys {synth_script}",
        design_logs / "synthesis.log",
        "Yosys Synthesis")

    return synth_ok


def main():
    parser = argparse.ArgumentParser(
        description="FlowMind P&R Flow (OpenROAD)")
    parser.add_argument("--design", required=True)
    parser.add_argument("--utilization", type=float, default=None,
                        help="Core utilization 0.0-1.0 (overrides config)")
    parser.add_argument("--skip-synth", action="store_true",
                        help="Skip synthesis (use existing netlist)")
    args = parser.parse_args()

    print("\n" + "="*55)
    print(f"  FlowMind P&R FLOW: {args.design}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    cfg = load_config(args.design, args.utilization)

    design_out  = OUTPUTS_DIR / args.design
    design_logs = LOGS_DIR    / args.design
    design_out.mkdir(parents=True,  exist_ok=True)
    design_logs.mkdir(parents=True, exist_ok=True)

    synth_netlist = design_out / f"{args.design}_synth.v"
    sdc_path      = design_out / f"{args.design}.sdc"
    pnr_netlist   = design_out / f"{args.design}_pnr.v"
    output_def    = design_out / f"{args.design}.def"
    route_guide   = design_out / f"{args.design}.guide"
    drc_report    = design_out / f"{args.design}_drc.rpt"
    pnr_script    = design_out / "pnr.tcl"
    pnr_log       = design_logs / "pnr.log"

    # Step 1: Synthesis
    if not args.skip_synth:
        synth_ok = run_synthesis_for_pnr(
            args.design, cfg, design_out, design_logs)
        if not synth_ok:
            print("ERROR: Synthesis failed.")
            sys.exit(1)
    else:
        print("\n[1/3] Skipping synthesis (--skip-synth)")
        if not synth_netlist.exists():
            print(f"ERROR: No netlist found at {synth_netlist}")
            sys.exit(1)

    # Step 2: Generate P&R script
    print("\n[2/3] Generating OpenROAD P&R script...")
    pnr_context = {
        **cfg,
        "timestamp":        datetime.now().isoformat(),
        "input_netlist":    str(synth_netlist),
        "input_sdc":        str(sdc_path),
        "output_netlist":   str(pnr_netlist),
        "output_def":       str(output_def),
        "output_route_guide": str(route_guide),
        "output_drc_report":  str(drc_report),
    }
    pnr_script.write_text(
        render_template("pnr.tcl.j2", pnr_context))
    print(f"  Written: {pnr_script}")

    # Step 3: Run OpenROAD
    print("\n[3/3] Running OpenROAD Place & Route...")
    pnr_ok = run_tool(
        f"openroad -exit {pnr_script}",
        pnr_log,
        "OpenROAD P&R")

    # Parse results
    print("\n" + "="*55)
    print("  P&R RESULTS")
    print("="*55)
    metrics = parse_pnr_log(pnr_log)

    print(f"  Status       : {metrics['status']}")
    if metrics['wns_ns'] is not None:
        print(f"  WNS          : {metrics['wns_ns']:.3f} ns")
    if metrics['tns_ns'] is not None:
        print(f"  TNS          : {metrics['tns_ns']:.3f} ns")
    if metrics['design_area'] is not None:
        print(f"  Design area  : {metrics['design_area']:.1f} um2")
    if metrics['drc_violations'] is not None:
        print(f"  DRC violations: {metrics['drc_violations']}")

    print(f"\n  Output DEF   : {output_def}")
    print(f"  Output netlist: {pnr_netlist}")
    print(f"  Log          : {pnr_log}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
