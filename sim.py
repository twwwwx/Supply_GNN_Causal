#!/usr/bin/env python3
from argparse import ArgumentParser
from pathlib import Path

from main import run_pipeline


def parse_sample_sizes(raw):
    values = [item.strip() for item in raw.split(",") if item.strip()]
    return [int(item) for item in values]


def main():
    parser = ArgumentParser()
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--sample-sizes", type=str, default="100,500,1000")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--metrics-csv", type=str, default="results/metrics.csv")
    args = parser.parse_args()

    sample_sizes = parse_sample_sizes(args.sample_sizes)
    root_dir = Path(__file__).resolve().parent
    metrics_csv = Path(args.metrics_csv)
    if not metrics_csv.is_absolute():
        metrics_csv = root_dir / metrics_csv

    rows = run_pipeline(args.num_runs, sample_sizes, args.seed, metrics_csv)
    print(f"Wrote {rows} metric rows to {metrics_csv}")


if __name__ == "__main__":
    main()
