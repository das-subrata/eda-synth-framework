"""
parse_qor.py
Parses Yosys synthesis and OpenSTA timing logs
and returns structured QoR metrics.
"""
import re


def parse_yosys_log(log_path: str) -> dict:
    """
    Extract area, cell count from Yosys synthesis log.
    """
    metrics = {
        "cell_count":    None,
        "chip_area_um2": None,
        "seq_pct":       None,
        "status":        "UNKNOWN"
    }

    try:
        with open(log_path) as f:
            content = f.read()

        if "End of script" in content:
            metrics["status"] = "SUCCESS"
        else:
            metrics["status"] = "FAILED"
            return metrics

        # Number of cells: 5774
        m = re.search(r"Number of cells:\s+(\d+)", content)
        if m:
            metrics["cell_count"] = int(m.group(1))

        # Chip area for module '\picorv32': 77250.339200
        m = re.search(r"Chip area for module.*?:\s+([\d.]+)", content)
        if m:
            metrics["chip_area_um2"] = float(m.group(1))

        # of which used for sequential elements: 44782.950400 (57.97%)
        m = re.search(
            r"sequential elements:\s+[\d.]+\s+\(([\d.]+)%\)", content)
        if m:
            metrics["seq_pct"] = float(m.group(1))

    except FileNotFoundError:
        metrics["status"] = "LOG_NOT_FOUND"

    return metrics


def parse_sta_log(log_path: str) -> dict:
    """
    Extract WNS, TNS, critical path delay from OpenSTA log.
    """
    metrics = {
        "wns_ns":           None,
        "tns_ns":           None,
        "critical_path_ns": None,
        "status":           "UNKNOWN"
    }

    try:
        with open(log_path) as f:
            content = f.read()

        # wns 0.00
        m = re.search(r"^wns\s+([-\d.]+)", content, re.MULTILINE)
        if m:
            metrics["wns_ns"] = float(m.group(1))

        # tns 0.00
        m = re.search(r"^tns\s+([-\d.]+)", content, re.MULTILINE)
        if m:
            metrics["tns_ns"] = float(m.group(1))

        # "   12.36   data arrival time"
        # Number comes BEFORE the text in OpenSTA output
        m = re.search(
            r"^\s+([\d.]+)\s+data arrival time",
            content, re.MULTILINE)
        if m:
            metrics["critical_path_ns"] = float(m.group(1))

        if metrics["wns_ns"] is not None:
            if metrics["wns_ns"] >= 0:
                metrics["status"] = "TIMING_MET"
            else:
                metrics["status"] = "TIMING_VIOLATED"

    except FileNotFoundError:
        metrics["status"] = "LOG_NOT_FOUND"

    return metrics


def print_qor_summary(design_name: str,
                      synth_metrics: dict,
                      sta_metrics: dict,
                      clock_period_ns: float):
    """Print a clean QoR summary table."""

    # Calculate max frequency
    max_freq = None
    if sta_metrics["critical_path_ns"] is not None:
        margin = 0.8  # setup time + uncertainty
        max_period = sta_metrics["critical_path_ns"] + margin
        max_freq = 1000.0 / max_period

    print("\n" + "="*55)
    print(f"  QoR SUMMARY: {design_name}")
    print("="*55)
    print(f"  Synthesis status : {synth_metrics['status']}")
    print(f"  STA status       : {sta_metrics['status']}")
    print("-"*55)
    print(f"  Cell count       : {synth_metrics['cell_count']}")
    if synth_metrics['chip_area_um2']:
        print(f"  Chip area        : "
              f"{synth_metrics['chip_area_um2']:.1f} um2")
    if synth_metrics['seq_pct']:
        print(f"  Sequential cells : {synth_metrics['seq_pct']:.1f}%")
    print("-"*55)
    print(f"  Clock period     : {clock_period_ns} ns")
    if sta_metrics['wns_ns'] is not None:
        print(f"  WNS              : {sta_metrics['wns_ns']:.3f} ns")
    if sta_metrics['tns_ns'] is not None:
        print(f"  TNS              : {sta_metrics['tns_ns']:.3f} ns")
    if sta_metrics['critical_path_ns']:
        print(f"  Critical path    : "
              f"{sta_metrics['critical_path_ns']:.2f} ns")
    if max_freq:
        print(f"  Max frequency    : {max_freq:.1f} MHz")
    print("="*55 + "\n")
