from pathlib import Path
import csv

from dGC.gen_data import sample_data_undir
from dGC.baseline import DEFAULT_MODEL_SPECS




def evaluate_model(sample, model_spec):
    if isinstance(sample, dict):
        x_values = sample["X"]
        y_values = sample["Y"]
    else:
        x_values = [row[0] for row in sample]
        y_values = [row[1] for row in sample]

    sq_errors = []
    beta2 = model_spec.get("beta2", 0.0)
    for x, y in zip(x_values, y_values):
        y_hat = model_spec["intercept"] + model_spec["beta"] * x + beta2 * (x ** 2)
        sq_errors.append((y - y_hat) ** 2)
    return sum(sq_errors) / len(sq_errors)


def append_metric(metrics_csv: Path, row: dict):
    metrics_csv.parent.mkdir(parents=True, exist_ok=True)
    file_exists = metrics_csv.exists()
    fieldnames = [
        "run_id",
        "sample_size",
        "model_name",
        "intercept",
        "beta",
        "beta2",
        "metric_name",
        "metric_value",
    ]
    with metrics_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def run_pipeline(
    num_runs,
    sample_sizes,
    base_seed,
    metrics_csv: Path,
    model_specs=None,
):
    if model_specs is None:
        model_specs = DEFAULT_MODEL_SPECS

    written_rows = 0
    for run_id in range(1, num_runs + 1):
        for sample_size in sample_sizes:
            sample = sample_data_undir(sample_size, base_seed + 1000 * run_id + sample_size)
            for spec in model_specs:
                mse = evaluate_model(sample, spec)
                append_metric(
                    metrics_csv,
                    {
                        "run_id": run_id,
                        "sample_size": sample_size,
                        "model_name": spec["model_name"],
                        "intercept": spec["intercept"],
                        "beta": spec["beta"],
                        "beta2": spec.get("beta2", 0.0),
                        "metric_name": "mse",
                        "metric_value": round(mse, 6),
                    },
                )
                written_rows += 1
    return written_rows
