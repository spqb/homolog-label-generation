"""Regression checks for the integrated RR generation PCA comparison."""

import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "analysis"))
from analyze_generated_samples import plot_rr_generation_pca, validate_label_counts


class GeneratedSampleLabelTest(unittest.TestCase):
    def test_pca_outputs_and_prescribed_label_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            outputs_root = root / "outputs"
            models_root = root / "models"
            labels = ["PF00486", "PF00486", "PF01339", "PF01339", "PF04397", "PF04397"]
            sequences = ["ACDE", "ACDF", "DCDE", "DCDF", "FCDE", "FCDF"]
            natural = pd.DataFrame({"header": [f"n{i}" for i in range(6)],
                                    "sequence_align": sequences, "true_label": labels})
            union = outputs_root / "RR/t1_0.7/seed_1/datasets_for_rbm"
            union.mkdir(parents=True)
            for budget in (100, 1000, 5000):
                natural.to_csv(union / f"rbm_dataset_ntrain_{budget}.csv", index=False)
            model_case = models_root / "RR/t1_0.7/seed_1"
            for name in ["RBM_labels_all_true_full", *(
                f"RBM_labels_{budget}_{method}"
                for budget in (100, 1000) for method in ("onlytrain", "onehot", "rbm", "ankh")
            )]:
                folder = model_case / name
                folder.mkdir(parents=True)
                natural.to_csv(folder / "samples.true_labels.csv", index=False)

            output_dir = root / "figures"
            plot_rr_generation_pca("RR", "0.7", 1, outputs_root, models_root,
                                   output_dir, 20, False)
            self.assertEqual(len(list(output_dir.glob("*.pdf"))), 2)
            self.assertTrue(all(path.stat().st_size > 1000 for path in output_dir.glob("*.pdf")))

            values = (np.zeros((6, 2)), np.asarray(labels))
            with self.assertRaises(ValueError):
                validate_label_counts(values, (np.zeros((5, 2)), np.asarray(labels[:-1])),
                                      model_case / "samples.true_labels.csv")


if __name__ == "__main__":
    unittest.main()
