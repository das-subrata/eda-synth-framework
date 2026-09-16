"""
qor_database.py
===============
SQLite database for storing and querying QoR metrics
across multiple designs, runs, and tool versions.
"""
import sqlite3
from datetime import datetime
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "results" / "qor_runs.db"


def init_database():
    """Create database and tables if they don't exist."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS runs (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp       TEXT NOT NULL,
            design_name     TEXT NOT NULL,
            clock_period_ns REAL NOT NULL,
            cell_count      INTEGER,
            chip_area_um2   REAL,
            seq_pct         REAL,
            wns_ns          REAL,
            tns_ns          REAL,
            critical_path_ns REAL,
            max_freq_mhz    REAL,
            synth_status    TEXT,
            sta_status      TEXT
        )
    """)

    conn.commit()
    conn.close()
    print(f"  Database ready: {DB_PATH}")


def insert_run(design_name: str,
               clock_period_ns: float,
               synth_metrics: dict,
               sta_metrics: dict) -> int:
    """Insert a new run into the database. Returns the run ID."""
    init_database()

    # Calculate max frequency
    max_freq = None
    if sta_metrics.get("critical_path_ns"):
        margin = 0.8
        max_freq = 1000.0 / (sta_metrics["critical_path_ns"] + margin)

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        INSERT INTO runs (
            timestamp, design_name, clock_period_ns,
            cell_count, chip_area_um2, seq_pct,
            wns_ns, tns_ns, critical_path_ns, max_freq_mhz,
            synth_status, sta_status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        datetime.now().isoformat(),
        design_name,
        clock_period_ns,
        synth_metrics.get("cell_count"),
        synth_metrics.get("chip_area_um2"),
        synth_metrics.get("seq_pct"),
        sta_metrics.get("wns_ns"),
        sta_metrics.get("tns_ns"),
        sta_metrics.get("critical_path_ns"),
        max_freq,
        synth_metrics.get("status"),
        sta_metrics.get("status"),
    ))

    run_id = c.lastrowid
    conn.commit()
    conn.close()

    print(f"  Saved to database: run ID {run_id}")
    return run_id


def query_all_runs() -> list:
    """Return all runs from the database as a list of dicts."""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM runs ORDER BY timestamp DESC")
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def query_design_runs(design_name: str) -> list:
    """Return all runs for a specific design."""
    init_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute(
        "SELECT * FROM runs WHERE design_name=? ORDER BY timestamp",
        (design_name,))
    rows = [dict(row) for row in c.fetchall()]
    conn.close()
    return rows


def print_runs_table(runs: list):
    """Print runs as a formatted table."""
    if not runs:
        print("  No runs in database yet.")
        return

    try:
        from tabulate import tabulate
        headers = ["ID", "Design", "Period", "Cells",
                   "Area(um2)", "WNS(ns)", "TNS(ns)",
                   "MaxFreq(MHz)", "STA Status"]
        rows = []
        for r in runs:
            rows.append([
                r["id"],
                r["design_name"],
                f"{r['clock_period_ns']:.1f}",
                r["cell_count"] or "-",
                f"{r['chip_area_um2']:.0f}" if r["chip_area_um2"] else "-",
                f"{r['wns_ns']:.3f}"  if r["wns_ns"]  is not None else "-",
                f"{r['tns_ns']:.3f}"  if r["tns_ns"]  is not None else "-",
                f"{r['max_freq_mhz']:.1f}" if r["max_freq_mhz"] else "-",
                r["sta_status"] or "-",
            ])
        print(tabulate(rows, headers=headers, tablefmt="grid"))
    except ImportError:
        for r in runs:
            print(f"  [{r['id']}] {r['design_name']} "
                  f"period={r['clock_period_ns']}ns "
                  f"WNS={r['wns_ns']}ns "
                  f"status={r['sta_status']}")
