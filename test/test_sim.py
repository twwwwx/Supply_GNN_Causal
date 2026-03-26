import csv
import tempfile
import unittest
from pathlib import Path

from main import run_pipeline


class PipelineTest(unittest.TestCase):
    def test_pipeline_writes_rows(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            metrics_csv = Path(tmp_dir) / "metrics.csv"
            rows = run_pipeline(
                num_runs=1,
                sample_sizes=[10],
                base_seed=7,
                metrics_csv=metrics_csv,
            )
            self.assertEqual(rows, 2)
            with metrics_csv.open("r", encoding="utf-8") as handle:
                reader = csv.DictReader(handle)
                data = list(reader)
            self.assertEqual(len(data), 2)
            self.assertEqual(data[0]["metric_name"], "mse")


if __name__ == "__main__":
    unittest.main()
