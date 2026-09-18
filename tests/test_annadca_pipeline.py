"""Exercise the installed annaDCA CLI through the project's real stage scripts."""
import csv
from importlib.metadata import version
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class AnnaDCAPipelineTest(unittest.TestCase):
    def test_train_and_conditioned_sample(self):
        self.assertEqual(version('annadca'), '0.2.3')
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            datasets = root / 'datasets'
            models = root / 'models'
            datasets.mkdir()
            rows = [
                {'header': f's{i}', 'sequence_align': seq,
                 'true_label': label, 'ankh_label': label,
                 'train_only': label if i < 2 else ''}
                for i, (seq, label) in enumerate([
                    ('ACDE', 'alpha'), ('ACDF', 'beta'),
                    ('ACGE', 'alpha'), ('ACGF', 'beta'),
                    ('DCDE', 'alpha'), ('DCDF', 'beta'),
                    ('DCGE', 'alpha'), ('DCGF', 'beta'),
                ])
            ]
            with (datasets / 'rbm_dataset_ntrain_4.csv').open('w') as f:
                writer = csv.DictWriter(f, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)

            # Retain the stage's actual CLI flags and paths, reducing only compute
            # budgets and selecting CPU so this test requires no GPU or model data.
            wrapper = root / 'annadca-smoke'
            wrapper.write_text(f'''#!{sys.executable}
import subprocess
import sys
args = sys.argv[1:]
if args[0] == "train":
    args += ["--hidden", "4", "--nepochs", "2", "--nchains", "4", "--no-triton"]
args += ["--device", "cpu", "--gibbs_steps", "2", "--seed", "7"]
sys.exit(subprocess.call([sys.executable, "-m", "annadca.cli", *args]))
''')
            wrapper.chmod(0o755)
            for stage in ['06_train_conditioned_rbms.sh', '07_sample_conditioned_rbms.sh']:
                command = [
                    'bash', str(ROOT / 'src/pipeline/stages' / stage),
                    '--datasets_dir', str(datasets), '--model_dir', str(models),
                    '--annadca_bin', str(wrapper),
                ]
                if stage.startswith('07_'):
                    command += ['--self_consistency_per_label', '2']
                result = subprocess.run(
                    command,
                    cwd=ROOT, text=True, capture_output=True, timeout=120,
                    env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

            for folder in ['RBM_labels_all_true_full', 'RBM_labels_4_onlytrain', 'RBM_labels_4_ankh']:
                with h5py.File(models / folder / 'params.h5') as f:
                    self.assertEqual(set(json.loads(f.attrs['label_vocabulary'])), {'alpha', 'beta'})
                    def check_finite(name, node):
                        if isinstance(node, h5py.Dataset) and node.dtype.kind == 'f':
                            self.assertTrue(np.isfinite(node[()]).all(), name)
                    f.visititems(check_finite)
                with (models / folder / 'samples.true_labels.csv').open() as f:
                    samples = list(csv.DictReader(f))
                self.assertEqual(len(samples), len(rows))
                self.assertEqual([r['true_label'] for r in samples], [r['true_label'] for r in rows])
                self.assertTrue(all(len(r['sequence_align']) == 4 for r in samples))
                self.assertTrue(all(set(r['sequence_align']) <= set('-ACDEFGHIKLMNPQRSTVWY') for r in samples))
                if folder != 'RBM_labels_4_onlytrain':
                    with (models / folder / 'samples.self_consistency.csv').open() as f:
                        self_samples = list(csv.DictReader(f))
                    self.assertEqual(len(self_samples), 4)
                    self.assertEqual(set(r['target_label'] for r in self_samples), {'alpha', 'beta'})

            posterior = models / 'posterior-smoke'
            result = subprocess.run(
                [sys.executable, '-m', 'annadca', 'train',
                 '-d', str(datasets / 'rbm_dataset_ntrain_4.csv'), '-o', str(posterior),
                 '--column_names', 'header', '--column_sequences', 'sequence_align',
                 '--column_labels', 'train_only', '--hidden', '4', '--nepochs', '2',
                 '--nchains', '4', '--gibbs_steps', '2', '--seed', '7', '--device', 'cpu',
                 '--no-triton', '--no_reweighting', '--marginalize_missing_labels'],
                cwd=ROOT, text=True, capture_output=True, timeout=120,
                env={**os.environ, 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertTrue((posterior / 'params.h5').is_file())


if __name__ == '__main__':
    unittest.main()
