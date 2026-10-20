#!/usr/bin/env python3
"""Compare paired conditional-generation runs; no PLM inference required."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, pdist
from sklearn.decomposition import PCA

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from analyze_generated_samples import connected_correlation_statistics, connected_correlation_pearson
from adabmDCA.fasta import encode_sequence, get_tokens

HISTOGRAM_PSEUDOCOUNT = 1e-10


def symmetric_kl(p, q, eps=HISTOGRAM_PSEUDOCOUNT):
    """Symmetric KL on smoothed, normalized PCA histograms."""
    p = p.ravel().astype(float) + eps
    q = q.ravel().astype(float) + eps
    p /= p.sum()
    q /= q.sum()
    return .5 * (np.sum(p * np.log(p / q)) + np.sum(q * np.log(q / p)))


def nearest_identity(query, reference):
    best = np.zeros(len(query))
    for i in range(0, len(query), 128):
        block = query[i:i + 128]
        values = np.zeros(len(block))
        for j in range(0, len(reference), 2048):
            values = np.maximum(values, 1 - cdist(block, reference[j:j + 2048], metric='hamming').min(axis=1))
        best[i:i + len(block)] = values
    return best


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir', type=Path, required=True)
    p.add_argument('--tables-dir', type=Path, required=True)
    p.add_argument('--images-dir', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[1, 2, 3])
    p.add_argument('--bins', type=int, default=20)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args()
    if a.bins < 2 or a.threads < 1:
        p.error('Require bins >= 2 and threads >= 1')
    import torch
    torch.set_num_threads(a.threads)
    tokens = get_tokens('protein')
    rows, timings = [], []
    for seed in a.seeds:
        case = a.run_dir / f'seed_{seed}'
        manifest = json.loads((case / 'manifest.json').read_text())
        data = Path(manifest['data'])
        if hashlib.sha256(data.read_bytes()).hexdigest() != manifest['data_sha256']:
            raise ValueError(f'Reference changed: {data}')
        natural = pd.read_csv(data, keep_default_na=False)
        encoded = encode_sequence(natural.sequence_align.tolist(), tokens)
        onehot = np.eye(len(tokens), dtype=np.float32)[encoded].reshape(len(encoded), -1)
        pca = PCA(n_components=2, svd_solver='randomized', random_state=0)
        projected = pca.fit_transform(onehot)
        del onehot
        frames, codes, projections = {}, {}, {}
        for arm in ['zero', 'posterior']:
            for stage in ['train', 'sample']:
                marker = case / f'{arm}.{stage}.complete.json'
                completed = json.loads(marker.read_text()) if marker.exists() else {}
                timings.append(dict(seed=seed, arm=arm, stage=stage, seconds=completed.get('seconds', np.nan)))
            frame = pd.read_csv(case / arm / 'samples.csv', keep_default_na=False)
            counts = frame.target.value_counts().to_dict()
            if counts != {k: manifest['samples_per_label'] for k in manifest['labels']}:
                raise ValueError(f'Unexpected conditioning labels/counts: {case / arm}')
            frames[arm] = frame
            codes[arm] = encode_sequence(frame.sequence_align.tolist(), tokens)
            projections[arm] = pca.transform(np.eye(len(tokens), dtype=np.float32)[codes[arm]].reshape(len(frame), -1))
        # One grid per split, shared across classes and both arms; no discarded outliers.
        combined = np.concatenate([projected, *projections.values()])
        edges = [np.linspace(combined[:, j].min() - 1e-6, combined[:, j].max() + 1e-6, a.bins + 1) for j in range(2)]
        for label in manifest['labels']:
            mask = natural.true_label.to_numpy() == label
            stats = connected_correlation_statistics(natural.loc[mask, 'sequence_align'].tolist(), tokens)
            hist = np.histogram2d(*projected[mask].T, bins=edges)[0]
            for arm in ['zero', 'posterior']:
                selected = frames[arm].target.to_numpy() == label
                seqs = frames[arm].loc[selected, 'sequence_align'].tolist()
                query = codes[arm][selected]
                identity = nearest_identity(query, encoded)
                rows.append(dict(seed=seed, arm=arm, label=label, n_natural=int(mask.sum()), n_generated=len(seqs),
                                 correlation=connected_correlation_pearson(stats, seqs, tokens),
                                 symmetric_kl=symmetric_kl(hist, np.histogram2d(*projections[arm][selected].T, bins=edges)[0]),
                                 unique_fraction=len(set(seqs)) / len(seqs),
                                 mean_pairwise_identity=float(1 - pdist(query, metric='hamming').mean()) if len(query) > 1 else np.nan,
                                 nearest_natural_identity=float(identity.mean()),
                                 exact_match_fraction=float((identity == 1).mean())))
        print(f'Analyzed split {seed}', flush=True)
    out = a.tables_dir
    out.mkdir(parents=True, exist_ok=True)
    a.images_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    metrics = ['correlation', 'symmetric_kl', 'unique_fraction', 'mean_pairwise_identity', 'nearest_natural_identity', 'exact_match_fraction']
    df.to_csv(out / 'by_label.csv', index=False)
    # Do not silently macro-average over undefined class metrics.
    macro = df.groupby(['seed', 'arm'])[metrics].agg(lambda x: x.mean(skipna=False)).reset_index()
    macro.to_csv(out / 'by_seed.csv', index=False)
    macro.groupby('arm')[metrics].agg(['mean', 'std', 'count']).to_csv(out / 'summary.csv')
    delta = macro.set_index(['seed', 'arm'])[metrics].unstack('arm')
    pd.DataFrame({m: delta[m]['posterior'] - delta[m]['zero'] for m in metrics}).to_csv(out / 'paired_differences.csv')
    pd.DataFrame(timings).to_csv(out / 'runtime.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), constrained_layout=True)
    for ax, metric, title in zip(axes, metrics[:2], ['Label-wise correlation (higher is better)', 'Label-wise symmetric KL (lower is better)']):
        for seed in a.seeds:
            pair = macro[macro.seed == seed].set_index('arm').loc[['zero', 'posterior']]
            ax.plot([0, 1], pair[metric], 'o-', label=f'Split {seed}')
        ax.set_xticks([0, 1], ['Zero label', 'Posterior'])
        ax.set_title(title, fontsize=10)
        ax.set_ylabel('Macro-average across labels')
        ax.grid(axis='y', alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(a.images_dir / f'{a.run_dir.name}_paired_comparison.pdf')
    fig.savefig(a.images_dir / f'{a.run_dir.name}_paired_comparison.png', dpi=180)
    plt.close(fig)
    (out / 'analysis_config.json').write_text(json.dumps({'seeds': a.seeds, 'bins': a.bins, 'pca_seed': 0, 'histogram_pseudocount': HISTOGRAM_PSEUDOCOUNT, 'reference': 'same training CSV, grouped by true_label; transductive', 'identity': 'all aligned sites, including gaps', 'delta': 'posterior minus zero'}, indent=2))
    print(f'Results: {out.resolve()}')


if __name__ == '__main__':
    main()
