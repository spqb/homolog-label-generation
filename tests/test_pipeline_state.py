"""Artifact signatures respond to input and output changes."""
import sys
import tempfile
import tomllib
import unittest
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src/pipeline'))
from pipeline_state import inspect, record, signature
from run_experiments import execute, parser


class PipelineStateTest(unittest.TestCase):
    def test_signature_and_completed_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / 'source.csv'
            output = root / 'split.csv'
            source.write_text('header,label\na,True\n')
            expected = signature([source], {'seed': 1, 't1': '0.7'})
            self.assertEqual(inspect([output], output, expected), 'BUILD')
            output.write_text('header,label\na,True\n')
            record(output, expected, [output])
            self.assertEqual(inspect([output], output, expected), 'REUSE')
            output.write_text('header,label\na,False\n')
            self.assertEqual(inspect([output], output, expected), 'STALE')
            output.write_text('header,label\na,True\n')
            source.write_text('header,label\nb,True\n')
            changed = signature([source], {'seed': 1, 't1': '0.7'})
            self.assertEqual(inspect([output], output, changed), 'STALE')

    def test_empty_completion_marker_is_required(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sample = root / 'samples.csv'
            marker = root / '.samples.gibbs_1000.complete'
            sample.write_text('header,sequence_align,true_label\na,AC,A\n')
            marker.touch()
            record(sample, 'sample-signature', [sample, marker])
            self.assertEqual(inspect([sample, marker], sample, 'sample-signature'), 'REUSE')
            marker.unlink()
            self.assertEqual(inspect([sample, marker], sample, 'sample-signature'), 'BUILD')

    def test_manifest_uses_merged_cm_and_ankh(self):
        config = tomllib.loads((ROOT / 'src/pipeline/experiments.toml').read_text())
        self.assertEqual(config['datasets']['cm']['source_csv'], 'data/cm/cm_sequences_merged.csv')
        self.assertIn('ankh', config['defaults']['classification_methods'])
        self.assertEqual(config['defaults']['generation_methods'], ['true', 'onehot', 'rbm', 'ankh'])
        self.assertTrue(config['datasets']['RR']['generation_enabled'])
        for dataset in ('Globin', 'SH3', 'cm'):
            self.assertFalse(config['datasets'][dataset]['generation_enabled'])

    def test_main_dispatches_generation_only_for_rr(self):
        config = tomllib.loads((ROOT / 'src/pipeline/experiments.toml').read_text())
        options = parser()
        args = options.parse_args(['--profile', 'main', '--datasets', 'RR', 'Globin',
                                   '--t1', '0.7', '--seeds', '1', '--plan'])
        with patch('run_experiments.run_case') as run_case, patch('run_experiments.run_analysis') as run_analysis:
            execute(args, options, config)
        cases = {call.args[0]: call.args[5] for call in run_case.call_args_list}
        analyses = {call.args[0]: call.args[6] for call in run_analysis.call_args_list}
        self.assertTrue({5, 6, 7, 8, 9} <= set(cases['RR']))
        self.assertEqual(set(cases['Globin']) & {5, 6, 7, 8}, set())
        self.assertEqual(set(analyses['Globin']) & {5, 6, 7, 8}, set())
        self.assertIn(9, analyses['Globin'])

    def test_generation_profile_rejects_non_rr_dataset(self):
        config = tomllib.loads((ROOT / 'src/pipeline/experiments.toml').read_text())
        options = parser()
        args = options.parse_args(['--profile', 'generation', '--datasets', 'SH3', '--plan'])
        with self.assertRaises(SystemExit) as result:
            execute(args, options, config)
        self.assertEqual(result.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
