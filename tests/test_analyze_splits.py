"""Split diagnostics support families with more than eight label classes."""

import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/analysis'))
from analyze_splits import plot_pca_projection


class SplitPCATest(unittest.TestCase):
    def test_fifteen_labels_fit_in_adaptive_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            splits = root / 'splits'
            splits.mkdir()
            amino_acids = 'ACDEFGHIKLMNPQR'
            train = [{'sequence_align': residue + 'AA', 'label': f'class_{index}'}
                     for index, residue in enumerate(amino_acids)]
            test = [{'sequence_align': residue + 'CA', 'label': f'class_{index}'}
                    for index, residue in enumerate(amino_acids)]
            pd.DataFrame(train).to_csv(splits / 'train_100.csv', index=False)
            pd.DataFrame(test).to_csv(splits / 'test.csv', index=False)
            output = root / 'pca.pdf'

            plot_pca_projection(root, 100, output)

            self.assertGreater(output.stat().st_size, 0)


if __name__ == '__main__':
    unittest.main()
