"""Missing source annotations must not enter the reference classifier."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from prediction.predict_from_csv_true import _filter_labeled_training_rows


class TrainingLabelsTest(unittest.TestCase):
    def test_missing_rows_keep_sequences_headers_and_labels_aligned(self):
        seqs, headers, labels = _filter_labeled_training_rows(
            ['AA', 'AC', 'AD', 'AE', 'AF', 'AG'],
            ['a', 'b', 'c', 'd', 'e', 'f'],
            ['alpha', np.nan, '', 'beta', None, '  '],
        )
        self.assertEqual(seqs.tolist(), ['AA', 'AE'])
        self.assertEqual(headers.tolist(), ['a', 'd'])
        self.assertEqual(labels.tolist(), ['alpha', 'beta'])

    def test_false_and_zero_are_valid_labels(self):
        for labels in ([False, True], [0, 1]):
            _, _, retained = _filter_labeled_training_rows(['AA', 'AC'], ['a', 'b'], labels)
            self.assertEqual(retained.tolist(), labels)

    def test_insufficient_observed_classes(self):
        for labels in ([np.nan, None], ['alpha', np.nan]):
            with self.assertRaisesRegex(ValueError, 'observed'):
                _filter_labeled_training_rows(['AA', 'AC'], ['a', 'b'], labels)
