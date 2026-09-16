#!/usr/bin/env python3
"""
run_signoff.py
==============
Multi-corner STA signoff using OpenSTA with OCV derating.
Runs SS, TT, and FF corners and reports worst-case per corner.

Usage:
    python3 run_signoff.py --design gcd
    python3 run_signoff.py --design picorv32
"""
import argparse
import subprocess
import sys
import re
from datetime import datetime
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from qor_database import init_database

BASE_DIR      = Path(__file__).parent
CONFIGS_DIR   = BASE_DIR / "configs"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUTS_DIR   = BASE_DIR / "outputs"
LOGS_DIR      = BASE_DIR / "logs"
CORNERS_FILE  = CONFIGS_DIR / "corners.yaml"


def load_config(design_name: str) -> dict:
    config_path = CONFIGS_DIR / f"{design_name}.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_corners() -> list:
    with open(CORNERS_FILE) as f:
        return yaml.safe_load(f)["corners"]


def render_template(template_name: str, context: dict) -> str:
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    return env.get_template(template_name).render(**context)


def run_sta_corner(script_path: Path, log_path: Path,
                   corner_name: str) -> bool:
    print(f"\n  Running OpenSTA [{corner_name}]...")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            f"sta {script_path}",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True)
        for line in process.stdout:
            print("  " + line, end="")
            log_file.write(line)
        process.wait()
    return process.returncode == 0


def parse_corner_results(log_path: Path) -> dict:
    """Parse WNS and TNS from a corner STA log."""
    metrics = {"wns_ns": None, "tns_ns": None, "status": "UNKNOWN"}
    try:
        content = log_path.read_text()
        m = re.search(r"^wns\s+([-\d.]+)", content, re.MULTILINE)
        if m:
            metrics["wns_ns"] = float(m.group(1))
        m = re.search(r"^tns\s+([-\d.]+)", content, re.MULTILINE)
        if m:
            metrics["tns_ns"] = float(m.group(1))
        if metrics["wns_ns"] is not None:
            metrics["status"] = (
                "TIMING_MET" if metrics["wns_ns"] >= 0
                else "TIMING_VIOLATED")
    except FileNotFoundError:
        metrics["status"] = "LOG_NOT_FOUND"
    return metrics


def print_signoff_summary(design_name: str,
                          corner_results: list,
                          clock_period_ns: float):
    print("\n" + "="*60)
    print(f"  MCMM SIGNOFF SUMMARY: {design_name}")
    print(f"  Clock period: {clock_period_ns} ns")
    print("="*60)
    print(f"  {'Corner':<20} {'WNS (ns)':<12} {'TNS (ns)':<12} {'Status'}")
    print("  " + "-"*55)

    all_met = True
    for cr in corner_results:
        wns = f"{cr['wns_ns']:.3f}" if cr["wns_ns"] is not None else "N/A"
        tns = f"{cr['tns_ns']:.3f}" if cr["tns_ns"] is not None else "N/A"
        status = cr["status"]
        icon = "✓" if status == "TIMING_MET" else "✗"
        print(f"  {cr['corner']:<20} {wns:<12} {tns:<12} {icon} {status}")
        if status != "TIMING_MET":
            all_met = False

    print("="*60)
    overall = "ALL CORNERS MET" if all_met else "TIMING VIOLATIONS FOUND"
    print(f"  Overall: {overall}")
    print("="*60 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="FlowMind MCMM Signoff (OpenSTA)")
    parser.add_argument("--design", required=True)
    parser.add_argument("--use-pnr-netlist", action="store_true",
                        help="Use post-P&R netlist instead of synthesis netlist")
    args = parser.parse_args()

    print("\n" + "="*60)
    print(f"  FlowMind MCMM SIGNOFF: {args.design}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    cfg     = load_config(args.design)
    corners = load_corners()

    design_out  = OUTPUTS_DIR / args.design
    design_logs = LOGS_DIR    / args.design
    design_out.mkdir(parents=True,  exist_ok=True)
    design_logs.mkdir(parents=True, exist_ok=True)

    # Choose netlist: P&R or synthesis
    if args.use_pnr_netlist:
        netlist = design_out / f"{args.design}_pnr.v"
    else:
        netlist = design_out / f"{args.design}_netlist.v"

    if not netlist.exists():
        print(f"ERROR: Netlist not found: {netlist}")
        print("  Run synthesis first: python3 run_synthesis.py --design gcd")
        sys.exit(1)

    sdc_path = design_out / f"{args.design}.sdc"
    if not sdc_path.exists():
        # Generate SDC
        from jinja2 import Environment, FileSystemLoader
        env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
        sdc_path.write_text(
            env.get_template("constraints.sdc.j2").render(
                **cfg, timestamp=datetime.now().isoformat()))

    # Run each corner
    corner_results = []
    for corner in corners:
        corner_name = corner["name"]
        sta_script  = design_out / f"sta_{corner_name}.tcl"
        sta_log     = design_logs / f"signoff_{corner_name}.log"
        timing_rpt  = design_out  / f"timing_{corner_name}.rpt"

        # Render template
        context = {
            **cfg,
            "corner":               corner,
            "timestamp":            datetime.now().isoformat(),
            "input_netlist":        str(netlist),
            "input_sdc":            str(sdc_path),
            "output_timing_report": str(timing_rpt),
        }
        sta_script.write_text(
            render_template("mcmm_sta.tcl.j2", context))

        # Run
        ok = run_sta_corner(sta_script, sta_log, corner_name)

        # Parse
        metrics = parse_corner_results(sta_log)
        metrics["corner"] = corner_name
        corner_results.append(metrics)

    # Print summary
    print_signoff_summary(
        args.design, corner_results,
        cfg["clock_period_ns"])


if __name__ == "__main__":
    main()
