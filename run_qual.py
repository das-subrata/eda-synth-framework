#!/usr/bin/env python3
"""
run_qual.py
===========
Tool qualification regression suite for FlowMind.
Certifies a new tool configuration before releasing to design teams.

Runs all test designs at reference and candidate configurations,
compares PPA delta, checks reproducibility, generates HTML report.

Usage:
    python3 run_qual.py
    python3 run_qual.py --designs gcd picorv32
"""
import argparse
import subprocess
import sys
import time
import json
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import yaml

sys.path.insert(0, str(Path(__file__).parent / "scripts"))
from parse_qor import parse_yosys_log, parse_sta_log

BASE_DIR    = Path(__file__).parent
CONFIGS_DIR = BASE_DIR / "configs"
OUTPUTS_DIR = BASE_DIR / "outputs"
LOGS_DIR    = BASE_DIR / "logs"
QUAL_DIR    = BASE_DIR / "qual_results"


def load_qual_config() -> dict:
    with open(CONFIGS_DIR / "qual_config.yaml") as f:
        return yaml.safe_load(f)


def load_design_config(design_name: str) -> dict:
    with open(CONFIGS_DIR / f"{design_name}.yaml") as f:
        return yaml.safe_load(f)


@dataclass
class RunResult:
    design:       str
    config_name:  str
    run_id:       int
    wns_ns:       Optional[float] = None
    tns_ns:       Optional[float] = None
    area_um2:     Optional[float] = None
    power_mw:     Optional[float] = None
    cell_count:   Optional[int]   = None
    runtime_s:    float = 0.0
    synth_status: str = "UNKNOWN"
    sta_status:   str = "UNKNOWN"


def run_synthesis_config(design_name: str,
                         config_name: str,
                         effort: str,
                         run_id: int) -> RunResult:
    """Run synthesis at a specific effort level, return metrics."""
    result = RunResult(
        design=design_name,
        config_name=config_name,
        run_id=run_id)

    run_dir  = QUAL_DIR / design_name / config_name / f"run{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Temporarily override synth_effort in config
    design_cfg = load_design_config(design_name)
    design_cfg["synth_effort"] = effort

    # Write temp config
    import tempfile
    tmp_cfg = run_dir / "config_override.yaml"
    with open(tmp_cfg, 'w') as f:
        yaml.dump(design_cfg, f)

    start = time.time()

    # Run synthesis via subprocess
    cmd = (f"python3 {BASE_DIR}/run_synthesis.py "
           f"--design {design_name} "
           f"--period {design_cfg['clock_period_ns']}")

    log_path = run_dir / "synthesis.log"
    with open(log_path, 'w') as log_file:
        proc = subprocess.run(
            cmd, shell=True,
            stdout=log_file,
            stderr=subprocess.STDOUT)

    result.runtime_s = time.time() - start

    # Copy logs for parsing
    src_log = LOGS_DIR / design_name / "synthesis.log"
    if src_log.exists():
        import shutil
        shutil.copy(src_log, run_dir / "synthesis.log")

    # Parse results
    synth_metrics = parse_yosys_log(str(run_dir / "synthesis.log"))
    sta_log = LOGS_DIR / design_name / "sta.log"
    sta_metrics = parse_sta_log(str(sta_log)) if sta_log.exists() else {}

    result.wns_ns      = sta_metrics.get("wns_ns")
    result.tns_ns      = sta_metrics.get("tns_ns")
    result.area_um2    = synth_metrics.get("chip_area_um2")
    result.cell_count  = synth_metrics.get("cell_count")
    result.synth_status = synth_metrics.get("status", "UNKNOWN")
    result.sta_status   = sta_metrics.get("status", "UNKNOWN")

    print(f"    Run {run_id}: WNS={result.wns_ns} "
          f"area={result.area_um2:.0f}um2 "
          f"runtime={result.runtime_s:.1f}s")

    return result


def check_acceptance(ref_results: list,
                     cand_results: list,
                     criteria: dict) -> list:
    """
    Compare reference vs candidate results.
    Returns list of (check_name, passed, detail) tuples.
    """
    checks = []

    for ref, cand in zip(ref_results, cand_results):
        design = ref.design

        # 1. WNS regression check
        if ref.wns_ns is not None and cand.wns_ns is not None:
            delta = cand.wns_ns - ref.wns_ns
            passed = delta >= -criteria["max_wns_regression_ns"]
            checks.append((
                f"{design}: WNS delta",
                passed,
                f"ref={ref.wns_ns:.3f}ns cand={cand.wns_ns:.3f}ns "
                f"delta={delta:+.3f}ns "
                f"(limit: -{criteria['max_wns_regression_ns']}ns)"
            ))

        # 2. Area increase check
        if ref.area_um2 and cand.area_um2:
            pct = 100 * (cand.area_um2 - ref.area_um2) / ref.area_um2
            passed = pct <= criteria["max_area_increase_pct"]
            checks.append((
                f"{design}: Area delta",
                passed,
                f"ref={ref.area_um2:.0f} cand={cand.area_um2:.0f} "
                f"delta={pct:+.1f}% "
                f"(limit: +{criteria['max_area_increase_pct']}%)"
            ))

        # 3. Runtime check
        if ref.runtime_s > 0:
            pct = 100 * (cand.runtime_s - ref.runtime_s) / ref.runtime_s
            passed = pct <= criteria["max_runtime_increase_pct"]
            checks.append((
                f"{design}: Runtime delta",
                passed,
                f"ref={ref.runtime_s:.1f}s cand={cand.runtime_s:.1f}s "
                f"delta={pct:+.1f}% "
                f"(limit: +{criteria['max_runtime_increase_pct']}%)"
            ))

    return checks


def check_reproducibility(repro_results: list,
                           criteria: dict) -> list:
    """Check WNS variance across repeated runs."""
    checks = []
    by_design = {}
    for r in repro_results:
        by_design.setdefault(r.design, []).append(r)

    for design, runs in by_design.items():
        wns_values = [r.wns_ns for r in runs if r.wns_ns is not None]
        if len(wns_values) < 2:
            continue
        variance_ps = (max(wns_values) - min(wns_values)) * 1000
        passed = variance_ps <= criteria["max_wns_variance_ps"]
        checks.append((
            f"{design}: Reproducibility",
            passed,
            f"WNS range={variance_ps:.1f}ps across {len(runs)} runs "
            f"(limit: {criteria['max_wns_variance_ps']}ps)"
        ))
    return checks


def generate_html_report(qual_cfg: dict,
                         ref_results: list,
                         cand_results: list,
                         ppa_checks: list,
                         repro_checks: list) -> Path:
    """Generate HTML qualification report."""
    all_checks = ppa_checks + repro_checks
    passed = sum(1 for _, p, _ in all_checks if p)
    failed = len(all_checks) - passed
    overall = "PASS" if failed == 0 else "FAIL"
    color = "#28a745" if overall == "PASS" else "#dc3545"

    rows = ""
    for name, p, detail in all_checks:
        icon   = "✓" if p else "✗"
        bg     = "#d4edda" if p else "#f8d7da"
        status = "PASS" if p else "FAIL"
        rows += f"""
        <tr style="background:{bg}">
            <td>{icon} {name}</td>
            <td><b>{status}</b></td>
            <td style="font-family:monospace;font-size:0.85em">{detail}</td>
        </tr>"""

    design_rows = ""
    for ref, cand in zip(ref_results, cand_results):
        wns_delta = ""
        area_delta = ""
        if ref.wns_ns is not None and cand.wns_ns is not None:
            d = cand.wns_ns - ref.wns_ns
            wns_delta = f"({d:+.3f}ns)"
        if ref.area_um2 and cand.area_um2:
            p = 100*(cand.area_um2-ref.area_um2)/ref.area_um2
            area_delta = f"({p:+.1f}%)"

        design_rows += f"""
        <tr>
            <td><b>{ref.design}</b></td>
            <td>{f'{ref.wns_ns:.3f}' if ref.wns_ns is not None else 'N/A'}</td>
            <td>{f'{cand.wns_ns:.3f}' if cand.wns_ns is not None else 'N/A'} {wns_delta}</td>
            <td>{f'{ref.area_um2:.0f}' if ref.area_um2 else 'N/A'}</td>
            <td>{f'{cand.area_um2:.0f}' if cand.area_um2 else 'N/A'} {area_delta}</td>
            <td>{ref.runtime_s:.1f}s</td>
            <td>{cand.runtime_s:.1f}s</td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html>
<head>
<title>FlowMind Tool Qualification Report</title>
<meta charset="utf-8">
<style>
  body {{ font-family: -apple-system, sans-serif;
          margin: 0; padding: 20px; background: #f5f5f5; }}
  h1 {{ color: #1f77b4; border-bottom: 2px solid #1f77b4;
        padding-bottom: 8px; }}
  h2 {{ color: #444; margin-top: 30px; }}
  .verdict {{ font-size: 2em; font-weight: bold;
               color: {color}; margin: 20px 0; }}
  .meta {{ background: white; padding: 15px; border-radius: 8px;
            margin: 15px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  table {{ width: 100%; border-collapse: collapse;
            background: white; border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  th {{ background: #1f77b4; color: white; padding: 10px;
        text-align: left; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid #eee; }}
  .summary {{ display: grid; grid-template-columns: repeat(3,1fr);
               gap: 15px; margin: 15px 0; }}
  .card {{ background: white; border-radius: 8px; padding: 15px;
            text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .num {{ font-size: 2em; font-weight: bold; color: #1f77b4; }}
</style>
</head>
<body>
<h1>FlowMind Tool Qualification Report</h1>
<div class="meta">
  <b>Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br>
  <b>Reference:</b> {qual_cfg['qualification']['reference']['name']} —
    {qual_cfg['qualification']['reference']['description']}<br>
  <b>Candidate:</b> {qual_cfg['qualification']['candidate']['name']} —
    {qual_cfg['qualification']['candidate']['description']}<br>
  <b>Designs tested:</b> {', '.join(qual_cfg['qualification'].get('designs_tested', []))}
</div>

<div class="verdict">Overall: {overall}</div>

<div class="summary">
  <div class="card"><div class="num">{len(all_checks)}</div>
    <div>Total Checks</div></div>
  <div class="card"><div class="num" style="color:#28a745">{passed}</div>
    <div>Passed</div></div>
  <div class="card"><div class="num" style="color:#dc3545">{failed}</div>
    <div>Failed</div></div>
</div>

<h2>PPA Comparison</h2>
<table>
  <tr>
    <th>Design</th>
    <th>WNS Ref (ns)</th><th>WNS Cand (ns)</th>
    <th>Area Ref (µm²)</th><th>Area Cand (µm²)</th>
    <th>Runtime Ref</th><th>Runtime Cand</th>
  </tr>
  {design_rows}
</table>

<h2>Qualification Checks</h2>
<table>
  <tr><th>Check</th><th>Result</th><th>Detail</th></tr>
  {rows}
</table>
</body>
</html>"""

    report_path = QUAL_DIR / "qual_report.html"
    report_path.write_text(html)
    return report_path


def main():
    parser = argparse.ArgumentParser(
        description="FlowMind Tool Qualification Suite")
    parser.add_argument("--designs", nargs="+", default=None,
                        help="Designs to test (default: from qual_config.yaml)")
    parser.add_argument("--skip-repro", action="store_true",
                        help="Skip reproducibility testing (faster)")
    args = parser.parse_args()

    qual_cfg  = load_qual_config()
    criteria  = qual_cfg["acceptance_criteria"]
    designs   = args.designs or qual_cfg["test_designs"]
    ref_cfg   = qual_cfg["qualification"]["reference"]
    cand_cfg  = qual_cfg["qualification"]["candidate"]
    n_repro   = qual_cfg["reproducibility_runs"]

    QUAL_DIR.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("  FlowMind TOOL QUALIFICATION SUITE")
    print(f"  Reference : {ref_cfg['name']}")
    print(f"  Candidate : {cand_cfg['name']}")
    print(f"  Designs   : {', '.join(designs)}")
    print("="*60)

    ref_results  = []
    cand_results = []
    repro_results = []

    for design in designs:
        print(f"\n[{design}] Running reference configuration...")
        ref = run_synthesis_config(
            design, ref_cfg["name"], ref_cfg["synth_effort"], run_id=1)
        ref_results.append(ref)

        print(f"[{design}] Running candidate configuration...")
        cand = run_synthesis_config(
            design, cand_cfg["name"], cand_cfg["synth_effort"], run_id=1)
        cand_results.append(cand)

        if not args.skip_repro:
            print(f"[{design}] Reproducibility check "
                  f"({n_repro} runs)...")
            for i in range(2, n_repro + 1):
                r = run_synthesis_config(
                    design, cand_cfg["name"],
                    cand_cfg["synth_effort"], run_id=i)
                repro_results.append(r)
            repro_results.append(cand)

    # Store designs tested in config for report
    qual_cfg["qualification"]["designs_tested"] = designs

    # Run checks
    print("\n" + "="*60)
    print("  RUNNING ACCEPTANCE CHECKS...")
    print("="*60)

    ppa_checks  = check_acceptance(ref_results, cand_results, criteria)
    repro_checks = (check_reproducibility(repro_results, criteria)
                    if not args.skip_repro else [])

    all_checks = ppa_checks + repro_checks
    passed = sum(1 for _, p, _ in all_checks if p)
    failed = len(all_checks) - passed

    for name, p, detail in all_checks:
        icon = "✓" if p else "✗"
        print(f"  {icon} {name}")
        print(f"    {detail}")

    # Generate report
    report_path = generate_html_report(
        qual_cfg, ref_results, cand_results,
        ppa_checks, repro_checks)

    print("\n" + "="*60)
    overall = "PASS" if failed == 0 else "FAIL"
    print(f"  QUALIFICATION: {overall}")
    print(f"  {passed}/{len(all_checks)} checks passed")
    print(f"  Report: {report_path}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
