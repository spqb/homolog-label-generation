"""Balanced self-consistency targets and distinct analysis paths."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "analysis"))
from analyze_generated_samples import prediction_h5_path, sample_csv_path


ROOT = Path(__file__).resolve().parents[1]


class SelfConsistencyTargetsTest(unittest.TestCase):
    def test_balanced_targets_and_artifact_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "dataset.csv"
            output = root / "targets.csv"
            with dataset.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=["header", "sequence_align", "ankh_label"])
                writer.writeheader()
                writer.writerows([
                    {"header": "a", "sequence_align": "ACDE", "ankh_label": "alpha"},
                    {"header": "b", "sequence_align": "DCDF", "ankh_label": "beta"},
                    {"header": "c", "sequence_align": "ACDF", "ankh_label": ""},
                ])
            subprocess.run(
                [sys.executable, str(ROOT / "src/pipeline/prepare_self_consistency_targets.py"),
                 "--dataset", str(dataset), "--label-column", "ankh_label",
                 "--per-label", "3", "--output", str(output)],
                check=True, capture_output=True, text=True,
            )
            with output.open() as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 6)
            self.assertEqual([row["target_label"] for row in rows].count("alpha"), 3)
            self.assertEqual([row["target_label"] for row in rows].count("beta"), 3)

        models = Path("/models")
        natural = sample_csv_path(models, "RR", "0.7", 1, "ankh", 100)
        predictions = prediction_h5_path(models, "RR", "0.7", 1, "ankh", 100)
        self.assertEqual(natural.name, "samples.true_labels.csv")
        self.assertEqual(predictions.name, "samples.self_consistency.predictions.ankh.h5")


if __name__ == "__main__":
    unittest.main()
