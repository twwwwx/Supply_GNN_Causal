#!/usr/bin/env python3
import argparse
import csv
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple


CFG_COLS = [
    "model",
    "DGP",
    "num_runs",
    "n",
    "seed",
    "gen_graph",
    "tau_dir_true",
    "tau_in_true",
    "tau_out_true",
    "tau_tot_true",
    "features",
    "clip",
    "L",
    "output_dim",
    "variance_type",
    "variance_method",
    "bandwidth",
]

MSE_COLS = ["MSE_tau_dir", "MSE_tau_in", "MSE_tau_out", "MSE_tau_tot"]

_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def first_float(text: Optional[str]) -> Optional[float]:
    if text is None:
        return None
    match = _NUM_RE.search(str(text))
    if not match:
        return None
    return float(match.group(0))


def is_valid_row(row: Dict[str, str]) -> bool:
    model = (row.get("model") or "").strip()
    if model not in {"gnn", "dirgnn"}:
        return False
    for col in ("n", "seed", "L", "output_dim", "tau_dir_true", "tau_in_true", "tau_out_true"):
        if first_float(row.get(col)) is None:
            return False
    return True


def summarize_group(rows: List[Tuple[int, Dict[str, str]]]) -> Dict[str, Tuple[float, float]]:
    summary: Dict[str, Tuple[float, float]] = {}
    for col in MSE_COLS:
        vals = [first_float(r[col]) for _, r in rows]
        vals = [v for v in vals if v is not None]
        if vals:
            summary[col] = (min(vals), max(vals))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Strict compare of MSEs across identical recorded configs.")
    parser.add_argument("--csv", type=Path, default=Path("results/metrics.csv"))
    parser.add_argument("--min-spread", type=float, default=1e-9, help="Only show groups with MSE spread above this.")
    args = parser.parse_args()

    with args.csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    valid = [(i, r) for i, r in enumerate(rows, start=2) if is_valid_row(r)]
    invalid_count = len(rows) - len(valid)

    grouped: Dict[Tuple[str, ...], List[Tuple[int, Dict[str, str]]]] = defaultdict(list)
    for line_no, row in valid:
        key = tuple((row.get(c) or "").strip() for c in CFG_COLS)
        grouped[key].append((line_no, row))

    dup_groups = [(k, v) for k, v in grouped.items() if len(v) > 1]
    unstable = []
    for key, items in dup_groups:
        summary = summarize_group(items)
        spreads = {c: summary[c][1] - summary[c][0] for c in summary}
        if any(spread > args.min_spread for spread in spreads.values()):
            unstable.append((key, items, summary, spreads))

    print(f"CSV: {args.csv}")
    print(f"Total rows: {len(rows)}")
    print(f"Valid rows used: {len(valid)}")
    print(f"Filtered malformed/non-metric rows: {invalid_count}")
    print(f"Duplicate config groups: {len(dup_groups)}")
    print(f"Duplicate groups with MSE differences: {len(unstable)}")
    print("")

    for key, items, summary, spreads in unstable:
        cfg = {c: key[i] for i, c in enumerate(CFG_COLS)}
        print("CONFIG:", cfg)
        print("LINES:", [ln for ln, _ in items])
        for col in MSE_COLS:
            if col in summary:
                lo, hi = summary[col]
                spread = spreads[col]
                print(f"  {col}: min={lo:.6f} max={hi:.6f} spread={spread:.6f}")
        print("  likely_unrecorded_confounds: dir_dropout_rate, device/cuda nondeterminism, code version drift")
        print("")

    if not unstable:
        print("No MSE differences across identical recorded configs.")


if __name__ == "__main__":
    main()
