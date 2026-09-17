#!/usr/bin/env python3
"""
Extension 4a: ML-Assisted Log Triage
TF-IDF vectorization + cosine similarity search over EDA tool logs
Usage: python run_log_triage.py --query "WNS violation" --top 5
"""
import argparse, json, re, textwrap, pickle
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PROJECT_ROOT = Path(__file__).parent
LOG_DIRS = [
    "logs",
    "qual_results",
    "outputs",
    "timing-analysis/logs",
    "synth-flow/logs",
]
ROOT_LOG_PATTERNS = ["*.log", "*.rpt"]
CHUNK_LINES = 20
INDEX_PATH  = PROJECT_ROOT / ".triage_index.pkl"

_NOISE_RE = re.compile(
    r"/\S+/\S+"                             # absolute paths
    r"|\b0x[0-9a-fA-F]+\b"                  # hex addresses
    r"|\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}" # ISO timestamps
    r"|\[\d+\]"                              # bracketed numbers
)

def clean(text: str) -> str:
    text = _NOISE_RE.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()

def collect_logs():
    logs = []
    for d in LOG_DIRS:
        p = PROJECT_ROOT / d
        if not p.exists():
            continue
        for pattern in ("**/*.log", "**/*.rpt"):
            logs.extend(p.glob(pattern))
    for pattern in ROOT_LOG_PATTERNS:
        logs.extend(PROJECT_ROOT.glob(pattern))
    return sorted(set(logs))

def chunk_file(path: Path):
    lines = path.read_text(errors="replace").splitlines()
    for i in range(0, len(lines), CHUNK_LINES):
        raw = "\n".join(lines[i : i + CHUNK_LINES])
        cleaned = clean(raw)
        if len(cleaned) < 20:
            continue
        yield cleaned, {
            "source":     str(path.relative_to(PROJECT_ROOT)),
            "start_line": i + 1,
            "end_line":   min(i + CHUNK_LINES, len(lines)),
        }

def build_index(force: bool = False):
    if not force and INDEX_PATH.exists():
        with open(INDEX_PATH, "rb") as f:
            idx = pickle.load(f)
        print(f"[triage] Index loaded ({len(idx['chunks'])} chunks from {len(set(m['source'] for m in idx['meta']))} files).")
        return idx

    log_files = collect_logs()
    if not log_files:
        raise RuntimeError("No log files found.")

    chunks, meta = [], []
    for path in log_files:
        for text, m in chunk_file(path):
            chunks.append(text)
            meta.append(m)

    print(f"[triage] Vectorizing {len(chunks)} chunks from {len(log_files)} files...")
    vec = TfidfVectorizer(
        max_features=5000,
        stop_words="english",
        token_pattern=r"[a-zA-Z_][a-zA-Z0-9_]{2,}",
        sublinear_tf=True,
    )
    matrix = vec.fit_transform(chunks)  # sparse, instant

    # Top corpus keywords
    idf_scores = vec.idf_
    terms = vec.get_feature_names_out()
    top_kw = [t for _, t in sorted(zip(idf_scores, terms), reverse=True)[:10]]
    print(f"[triage] Top TF-IDF keywords: {', '.join(top_kw)}")

    idx = {"chunks": chunks, "meta": meta, "vectorizer": vec, "matrix": matrix}
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(idx, f)
    print(f"[triage] Index saved to {INDEX_PATH}")
    return idx

def query_index(idx, query: str, top_n: int = 5):
    vec    = idx["vectorizer"]
    matrix = idx["matrix"]
    q_vec  = vec.transform([clean(query)])
    scores = cosine_similarity(q_vec, matrix).flatten()
    top_i  = np.argsort(scores)[::-1][:top_n]
    hits = []
    for i in top_i:
        hits.append({
            "score":   round(float(scores[i]), 4),
            "source":  idx["meta"][i]["source"],
            "lines":   f"{idx['meta'][i]['start_line']}-{idx['meta'][i]['end_line']}",
            "excerpt": textwrap.shorten(idx["chunks"][i], width=300, placeholder="…"),
        })
    return hits

def main():
    ap = argparse.ArgumentParser(description="EDA log triage — TF-IDF search")
    ap.add_argument("--query",   default="WNS violation timing slack")
    ap.add_argument("--top",     type=int, default=5)
    ap.add_argument("--reindex", action="store_true")
    ap.add_argument("--out",     default="reports/triage_report.json")
    args = ap.parse_args()

    Path("reports").mkdir(exist_ok=True)
    idx  = build_index(force=args.reindex)
    hits = query_index(idx, args.query, args.top)

    report = {"query": args.query, "top_k": args.top, "results": hits}
    Path(args.out).write_text(json.dumps(report, indent=2))

    print(f'\n── Query: "{args.query}" ──────────────────────────────')
    for i, h in enumerate(hits, 1):
        print(f"\n[{i}] score={h['score']}  {h['source']}  lines {h['lines']}")
        print(f"    {h['excerpt']}")
    print(f"\nReport → {args.out}")

if __name__ == "__main__":
    main()
