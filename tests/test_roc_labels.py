"""Check class-column alignment and per-label ROC plotting."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import matplotlib
matplotlib.use("Agg")
from matplotlib.axes import Axes
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "analysis"))
from analyze_classification import per_label_roc, plot_roc_labels, plot_roc_summary
from common import load_prediction_file


class LabelROCTest(unittest.TestCase):
    def test_column_order_and_missing_class(self):
        truth = np.array(["a", "a", "c", "c"])
        classes = np.array(["c", "b", "a"])
        scores = np.array([[.1, .1, .8], [.2, .1, .7], [.8, .1, .1], [.7, .1, .2]])
        curves = per_label_roc(truth, scores, classes)
        self.assertEqual(set(curves), {"a", "c"})
        for _, tpr, auc in curves.values():
            self.assertEqual(auc, 1.0)
            self.assertTrue(np.isfinite(tpr).all())

    def test_binary_and_degenerate(self):
        curves = per_label_roc(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9]), np.array([0, 1]))
        self.assertEqual([curve[2] for curve in curves.values()], [1.0, 1.0])
        self.assertEqual(per_label_roc(np.array([0, 0]), np.array([[.8, .2], [.9, .1]]), np.array([0, 1])), {})
        with self.assertRaises(ValueError):
            per_label_roc(np.array([0, 1]), np.ones((2, 3)), np.array([0, 1]))

    def test_hdf5_mapping_and_single_seed_grid(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "RR/t1_0.4/seed_1/predictions"
            folder.mkdir(parents=True)
            path = folder / "test.embedding.ankh.ntrain_100.predictions.h5"
            with h5py.File(path, "w") as f:
                f["train/labels_true"] = np.array([b"c", b"b", b"a"])
                f["test/labels_true"] = np.array([b"a", b"a", b"c", b"c"])
                f["test/predictions/logreg/labels_pred"] = np.array([b"a", b"a", b"c", b"c"])
                f["test/predictions/logreg/labels_probs"] = [[.8, .1, .1], [.7, .1, .2], [.1, .1, .8], [.2, .1, .7]]
            loaded = load_prediction_file(path)
            self.assertEqual(loaded["class_labels"].tolist(), ["a", "b", "c"])
            out = root / "grid.png"
            plot_roc_labels("RR", "0.4", [1, 2], 100, ["ankh", "rbm"], "logreg", out, root)
            self.assertGreater(out.stat().st_size, 1000)

            # Both ROC views must remain usable when the CLI hides their bands.
            with patch.object(Axes, "fill_between", side_effect=AssertionError("unexpected ROC band")):
                plot_roc_labels("RR", "0.4", [1], 100, ["ankh"], "logreg",
                                root / "grid_no_band.png", root, show_error_band=False)
                plot_roc_summary("RR", "0.4", [1], 100, ["ankh"], "logreg",
                                 root / "summary_no_band.png", root, show_error_band=False)
            self.assertGreater((root / "grid_no_band.png").stat().st_size, 1000)
            self.assertGreater((root / "summary_no_band.png").stat().st_size, 1000)


if __name__ == "__main__":
    unittest.main()
