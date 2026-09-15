#!/usr/bin/env python3
"""
run_synthesis.py
================
Main EDA flow orchestration script.

Usage:
    python3 run_synthesis.py --design picorv32
    python3 run_synthesis.py --design picorv32 --period 15.0

What it does:
    1. Reads design config from configs/<design>.yaml
    2. Generates Yosys synthesis script from Jinja2 template
    3. Generates SDC constraints from Jinja2 template
    4. Generates OpenSTA script from Jinja2 template
    5. Runs Yosys synthesis
    6. Runs OpenSTA timing analysis
    7. Parses results and prints QoR summary
"""
import argparse
import subprocess
import sys
import os
from datetime import datetime
from pathlib import Path

import yaml
from jinja2 import Environment, FileSystemLoader

# Add scripts/ to path so we can import parse_qor
sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from parse_qor import parse_yosys_log, parse_sta_log, print_qor_summary


# ── Paths ──────────────────────────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
CONFIGS_DIR   = BASE_DIR / "configs"
TEMPLATES_DIR = BASE_DIR / "templates"
OUTPUTS_DIR   = BASE_DIR / "outputs"
LOGS_DIR      = BASE_DIR / "logs"


def load_config(design_name: str, clock_period_override: float = None) -> dict:
    """Load design YAML config and apply any overrides."""
    config_path = CONFIGS_DIR / f"{design_name}.yaml"
    if not config_path.exists():
        print(f"ERROR: Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    # Apply clock period override if provided
    if clock_period_override:
        cfg["clock_period_ns"] = clock_period_override
        print(f"  Clock period overridden to {clock_period_override} ns")

    return cfg


def render_template(template_name: str, context: dict) -> str:
    """Render a Jinja2 template with the given context."""
    env = Environment(loader=FileSystemLoader(str(TEMPLATES_DIR)))
    template = env.get_template(template_name)
    return template.render(**context)


def run_tool(command: str, log_path: Path, tool_name: str) -> bool:
    """
    Run an EDA tool command, stream output to screen and save to log.
    Returns True if successful, False if failed.
    """
    print(f"\n  Running {tool_name}...")
    print(f"  Command: {command}")
    print(f"  Log: {log_path}")
    print("  " + "-"*50)

    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as log_file:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )

        for line in process.stdout:
            print("  " + line, end="")
            log_file.write(line)

        process.wait()

    if process.returncode != 0:
        print(f"\n  ERROR: {tool_name} exited with code {process.returncode}")
        return False

    print(f"\n  {tool_name} completed successfully.")
    return True


def main():
    # ── Parse arguments ───────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        description="EDA Synthesis Flow Automation")
    parser.add_argument(
        "--design", required=True,
        help="Design name (must match configs/<design>.yaml)")
    parser.add_argument(
        "--period", type=float, default=None,
        help="Clock period in ns (overrides config)")
    parser.add_argument(
        "--skip-sta", action="store_true",
        help="Skip timing analysis, run synthesis only")
    args = parser.parse_args()

    # ── Setup ─────────────────────────────────────────────────────────────
    print("\n" + "="*55)
    print(f"  EDA SYNTHESIS FLOW: {args.design}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*55)

    cfg = load_config(args.design, args.period)

    # Create output and log directories for this design
    design_out  = OUTPUTS_DIR / args.design
    design_logs = LOGS_DIR    / args.design
    design_out.mkdir(parents=True,  exist_ok=True)
    design_logs.mkdir(parents=True, exist_ok=True)

    # Define output file paths
    output_netlist       = design_out / f"{args.design}_netlist.v"
    output_sdc           = design_out / f"{args.design}.sdc"
    output_sta_script    = design_out / f"run_sta.tcl"
    output_timing_report = design_out / f"timing_report.txt"
    synth_log            = design_logs / "synthesis.log"
    sta_log              = design_logs / "sta.log"

    # Build Jinja2 context
    context = {
        **cfg,
        "timestamp":            datetime.now().isoformat(),
        "output_netlist":       str(output_netlist),
        "output_sdc":           str(output_sdc),
        "output_timing_report": str(output_timing_report),
    }

    # ── Step 1: Generate synthesis script ─────────────────────────────────
    print("\n[1/4] Generating synthesis script...")
    synth_script_path = design_out / "synthesize.ys"
    synth_script_path.write_text(
        render_template("synthesize.ys.j2", context))
    print(f"  Written: {synth_script_path}")

    # ── Step 2: Generate SDC constraints ──────────────────────────────────
    print("\n[2/4] Generating SDC constraints...")
    output_sdc.write_text(
        render_template("constraints.sdc.j2", context))
    print(f"  Written: {output_sdc}")

    # ── Step 3: Run Yosys synthesis ───────────────────────────────────────
    print("\n[3/4] Running Yosys synthesis...")
    synth_ok = run_tool(
        command=f"yosys {synth_script_path}",
        log_path=synth_log,
        tool_name="Yosys"
    )

    if not synth_ok:
        print("\nERROR: Synthesis failed. Check log:")
        print(f"  {synth_log}")
        sys.exit(1)

    # ── Step 4: Run OpenSTA timing analysis ───────────────────────────────
    if not args.skip_sta:
        print("\n[4/4] Running OpenSTA timing analysis...")

        # Generate STA script
        context["output_sdc"] = str(output_sdc)
        output_sta_script.write_text(
            render_template("run_sta.tcl.j2", context))

        sta_ok = run_tool(
            command=f"sta {output_sta_script}",
            log_path=sta_log,
            tool_name="OpenSTA"
        )
    else:
        print("\n[4/4] Skipping STA (--skip-sta flag set)")
        sta_ok = False

    # ── Parse and print QoR summary ───────────────────────────────────────
    print("\n" + "="*55)
    print("  PARSING RESULTS...")
    print("="*55)

    synth_metrics = parse_yosys_log(str(synth_log))
    sta_metrics   = parse_sta_log(str(sta_log)) if sta_ok else {
        "wns_ns": None, "tns_ns": None,
        "critical_path_ns": None, "status": "SKIPPED"
    }

    print_qor_summary(
        args.design,
        synth_metrics,
        sta_metrics,
        cfg["clock_period_ns"]
    )

    print(f"  Output files in: {design_out}")
    print(f"  Log files in:    {design_logs}\n")


if __name__ == "__main__":
    main()
