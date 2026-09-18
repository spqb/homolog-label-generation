#!/usr/bin/env python3
"""Run matched zero-label/posterior RR experiments with the locked annaDCA release."""
import argparse
import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def nonempty(path):
    return path.is_file() and path.stat().st_size > 0


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--outputs-root', type=Path, default=ROOT / 'outputs')
    p.add_argument('--run-dir', type=Path, default=None)
    p.add_argument('--models-root', type=Path, default=ROOT / 'models' / 'missing_labels')
    p.add_argument('--dataset', default='RR')
    p.add_argument('--t1', default='0.4')
    p.add_argument('--train-size', type=int, default=100)
    p.add_argument('--seeds', type=int, nargs='+', default=[1, 2, 3])
    for name, default in [('updates', 10000), ('hidden', 500), ('nchains', 5000),
                          ('gibbs-steps', 10), ('sample-steps', 1000), ('samples-per-label', 1000)]:
        p.add_argument('--' + name, type=int, default=default)
    p.add_argument('--lr', type=float, default=.005)
    p.add_argument('--device', default='cuda')
    p.add_argument('--triton', action=argparse.BooleanOptionalAction, default=True,
                   help='Use Triton for both training arms (default); --no-triton selects PyTorch.')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--skip-analysis', action='store_true')
    a = p.parse_args()
    if a.run_dir is None:
        a.run_dir = a.outputs_root / 'missing_labels' / a.dataset / f't1_{a.t1}' / f'ntrain_{a.train_size}'
    if any(getattr(a, k) <= 0 for k in ['updates', 'hidden', 'nchains', 'gibbs_steps', 'sample_steps', 'samples_per_label', 'train_size']) or a.lr <= 0:
        p.error('Budgets, sizes, and learning rate must be positive')
    try:
        annadca_version = version('annadca')
    except PackageNotFoundError:
        p.error('annaDCA is not installed; run uv sync --locked')
    if annadca_version != '0.2.3':
        p.error(f'annaDCA 0.2.3 is required, found {annadca_version}; run uv sync --locked')
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    env.pop('PYTHONPATH', None)
    import pandas as pd
    for seed in a.seeds:
        case = a.run_dir.resolve() / f'seed_{seed}'
        # Existing results retain their original manifest, even if CLI defaults or
        # the installed version has changed since training. Analysis validates the reference.
        if nonempty(case / 'manifest.json') and all(
            nonempty(a.models_root / a.dataset / f't1_{a.t1}' / f'ntrain_{a.train_size}' / f'seed_{seed}' / arm / 'params.h5')
            and nonempty(case / arm / 'samples.csv')
            for arm in ['zero', 'posterior']
        ):
            print(f'Reusing existing models and samples for split {seed}.', flush=True)
            continue
        data = a.outputs_root.resolve() / a.dataset / f't1_{a.t1}' / f'seed_{seed}' / 'datasets_for_rbm' / f'rbm_dataset_ntrain_{a.train_size}.csv'
        df = pd.read_csv(data, keep_default_na=False)
        if not {'header', 'sequence_align', 'train_only', 'true_label'} <= set(df):
            raise ValueError(f'Missing required columns: {data}')
        labels = sorted(set(df.train_only) - {''})
        if not labels or set(labels) != set(df.true_label) or not (df.train_only == '').any():
            raise ValueError('Require missing labels and all reference classes represented among labeled rows')
        if df.sequence_align.str.len().nunique() != 1:
            raise ValueError('Sequences must be aligned')
        config = {k: v for k, v in vars(a).items() if k not in ['seeds', 'dry_run', 'skip_analysis']}
        config.update(seed=seed, data=str(data), data_sha256=digest(data), annadca_version=annadca_version,
                      labeled_count=int((df.train_only != '').sum()), total_count=len(df), labels=labels)
        config = json.loads(json.dumps(config, default=str))
        manifest = case / 'manifest.json'
        if manifest.exists() and json.loads(manifest.read_text()) != config:
            raise ValueError(f'Configuration or source changed; use a new --run-dir: {case}')
        if not a.dry_run:
            case.mkdir(parents=True, exist_ok=True)
            manifest.write_text(json.dumps(config, indent=2) + '\n')
            # Sequences provide dimensions/alphabet only; sampler initializes its own chains.
            pd.DataFrame([{'header': f'{k}_{i}', 'sequence_align': df.sequence_align.iloc[0], 'target': k}
                          for k in labels for i in range(a.samples_per_label)]).to_csv(case / 'targets.csv', index=False)
        for arm in ['zero', 'posterior']:
            model = a.models_root / a.dataset / f't1_{a.t1}' / f'ntrain_{a.train_size}' / f'seed_{seed}' / arm
            sample_file = case / arm / 'samples.csv'
            common = ['--column_names', 'header', '--column_sequences', 'sequence_align', '--device', a.device, '--seed', str(seed)]
            train = [sys.executable, '-m', 'annadca', 'train', '-d', str(data), '-o', str(model),
                     '--column_labels', 'train_only', '--nepochs', str(a.updates), '--hidden', str(a.hidden),
                     '--lr', str(a.lr), '--nchains', str(a.nchains), '--gibbs_steps', str(a.gibbs_steps),
                     '--no_reweighting', '--triton' if a.triton else '--no-triton', *common]
            if arm == 'posterior':
                train.append('--marginalize_missing_labels')
            sample = [sys.executable, '-m', 'annadca', 'sample', '-d', str(case / 'targets.csv'),
                      '-p', str(model / 'params.h5'), '-o', str(sample_file),
                      '--column_labels', 'target', '--gibbs_steps', str(a.sample_steps), *common]
            for stage, command in [('train', train), ('sample', sample)]:
                marker = case / f'{arm}.{stage}.complete.json'
                artifact = model / 'params.h5' if stage == 'train' else sample_file
                if nonempty(artifact):
                    print(f'Reusing existing {seed}/{arm}/{stage}: {artifact}', flush=True)
                    continue
                print(shlex.join(command), flush=True)
                if a.dry_run:
                    continue
                if stage == 'sample':
                    sample_file.parent.mkdir(parents=True, exist_ok=True)
                if stage == 'train' and model.exists():
                    raise RuntimeError(f'Incomplete training: {model}. Preserve it elsewhere or use a new run directory.')
                start = time.monotonic()
                with (case / f'{arm}.{stage}.log').open('wb') as log:
                    with subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE,
                                          stderr=subprocess.STDOUT) as process:
                        try:
                            while chunk := process.stdout.read1(8192):
                                sys.stdout.buffer.write(chunk)
                                sys.stdout.buffer.flush()
                                log.write(chunk)
                                log.flush()
                            returncode = process.wait()
                        except BaseException:
                            process.terminate()
                            process.wait()
                            raise
                        if returncode:
                            raise subprocess.CalledProcessError(returncode, command)
                if not artifact.is_file() or not artifact.stat().st_size:
                    raise RuntimeError(f'Missing output: {artifact}')
                marker.write_text(json.dumps({'seconds': time.monotonic() - start, 'command': command}, indent=2))
    if not a.skip_analysis:
        command = ['bash', str(ROOT / 'src/pipeline/stages/09_analyze.sh'), 'missing-labels',
                   '--run-dir', str(a.run_dir.resolve()), '--seeds', *map(str, a.seeds),
                   '--tables-dir', str(a.outputs_root / 'analysis' / 'missing_labels' / a.dataset / f't1_{a.t1}' / f'ntrain_{a.train_size}' / 'tables'),
                   '--images-dir', str(a.outputs_root / 'images' / 'missing_labels' / a.dataset / f't1_{a.t1}')]
        print(shlex.join(command), flush=True)
        if not a.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == '__main__':
    main()
