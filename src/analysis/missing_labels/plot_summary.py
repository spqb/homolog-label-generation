#!/usr/bin/env python3
"""Publication-style summary of the two missing-label experiments."""
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/missing-label-mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'outputs' / 'missing_labels' / 'RR' / 't1_0.4'
COLORS = {'zero': '#737E8C', 'posterior': '#087F8C'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, default=[BASE / 'ntrain_100', BASE / 'ntrain_1000'])
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'outputs' / 'analysis' / 'missing_labels' / 'RR' / 't1_0.4' / 'combined' / 'tables')
    parser.add_argument('--images-dir', type=Path, default=ROOT / 'outputs' / 'images' / 'missing_labels' / 'RR' / 't1_0.4')
    args = parser.parse_args()
    frames, counts = [], {}
    for run in args.runs:
        if run.parent.parent.parent.name == 'missing_labels':
            by_seed = run.parent.parent.parent.parent / 'analysis' / 'missing_labels' / run.parent.parent.name / run.parent.name / run.name / 'tables' / 'by_seed.csv'
        else:
            by_seed = run / 'analysis' / 'by_seed.csv'  # legacy run layout
        frame = pd.read_csv(by_seed)
        if frame.duplicated(['seed', 'arm']).any():
            raise ValueError(f'Duplicate seed/arm: {run}')
        sizes = set()
        actual = []
        for seed, group in frame.groupby('seed'):
            if set(group.arm) != {'zero', 'posterior'}:
                raise ValueError(f'Incomplete pair: {run}, seed {seed}')
            meta = json.loads((run / f'seed_{seed}/manifest.json').read_text())
            sizes.add(meta['train_size'])
            actual.append(meta['labeled_count'])
        if len(sizes) != 1:
            raise ValueError(f'Mixed training sizes: {run}')
        size = sizes.pop()
        counts[size] = actual
        frame['train_size'] = size
        frames.append(frame)
    df = pd.concat(frames, ignore_index=True)
    sizes = sorted(df.train_size.unique())
    if df.duplicated(['train_size', 'seed', 'arm']).any():
        raise ValueError('Repeated training sizes across inputs')
    metrics = ['correlation', 'symmetric_kl']
    if not np.isfinite(df[metrics]).all().all():
        raise ValueError('Nonfinite primary metrics; inspect by-label results')
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.labelcolor': '#273440',
                         'text.color': '#273440', 'xtick.color': '#52606D', 'ytick.color': '#52606D',
                         'axes.spines.top': False, 'axes.spines.right': False, 'axes.spines.left': False,
                         'axes.edgecolor': '#CED5DC', 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 4.1))
    fig.subplots_adjust(left=.085, right=.975, bottom=.20, top=.77, wspace=.30)
    summaries = []
    for ax, metric, title in zip(axes, metrics,
        ['Correlation recovery ↑', 'Symmetric KL divergence ↓']):
        ax.set_title(title, fontsize=12, pad=12, weight='bold')
        for i, size in enumerate(sizes):
            sub = df[df.train_size == size]
            pivot = sub.pivot(index='seed', columns='arm', values=metric).sort_index()
            offsets = np.linspace(-.04, .04, len(pivot))
            for offset, (seed, row) in zip(offsets, pivot.iterrows()):
                x = np.array([i - .17, i + .17]) + offset
                ax.plot(x, [row.zero, row.posterior], color='#B8C2CC', lw=1.2, zorder=2)
                for xpos, arm in zip(x, ['zero', 'posterior']):
                    ax.scatter(xpos, row[arm], s=39, color=COLORS[arm], edgecolor='white', linewidth=.8, zorder=3)
            for x, arm in [(i - .17, 'zero'), (i + .17, 'posterior')]:
                mean = pivot[arm].mean()
                ax.plot([x - .08, x + .08], [mean, mean], color=COLORS[arm], lw=3.2, solid_capstyle='round', zorder=4)
                summaries.append(dict(train_size=size, metric=metric, arm=arm, mean=mean, std=pivot[arm].std(), n=len(pivot)))
        ax.set_xlim(-.48, len(sizes) - .52)
        lo, hi = df[metric].min(), df[metric].max()
        ax.set_ylim(lo - .20 * (hi - lo), hi + .18 * (hi - lo))
        ax.set_xticks(range(len(sizes)), [f'{n:,}' for n in sizes])
        ax.set_xlabel('Labeled sequences (nominal)', labelpad=9)
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.grid(axis='y', color='#E8EDF1', linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', length=0, pad=7)
    fig.legend(handles=[Line2D([], [], marker='o', color=COLORS[a], lw=0, markersize=6, label=l)
                        for a,l in [('zero','Zero label'),('posterior','Posterior')]] +
                       [Line2D([], [], color='#273440', lw=3, label='Mean')],
               loc='upper center', bbox_to_anchor=(.53,.99), frameon=False, ncol=3,
               columnspacing=2, handlelength=1.4, fontsize=10)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.images_dir.mkdir(parents=True, exist_ok=True)
    for extension in ['png', 'pdf', 'svg']:
        fig.savefig(args.images_dir / f'missing_labels_summary.{extension}', dpi=300, facecolor='white')
    plt.close(fig)
    count_text = '; '.join(f'{size:,}: ' + ', '.join(map(str, counts[size])) for size in sizes)
    (args.output_dir / 'figure_caption.txt').write_text(
        'Comparison of zero-label treatment and posterior marginalization. '
        'Each point is a macro-average across eight labels for one seed; lines connect paired runs, '
        'and horizontal bars show the mean across seeds. Higher correlation recovery and lower '
        'symmetric KL divergence indicate better distribution recovery. '
        f'Actual labeled counts by seed (nominal budget: counts): {count_text}. '
        'Evaluation uses the natural training sequence collection.\n')
    df.to_csv(args.output_dir / 'combined_by_seed.csv', index=False)
    pd.DataFrame(summaries).to_csv(args.output_dir / 'summary_statistics.csv', index=False)
    print(pd.DataFrame(summaries).to_string(index=False))
    print(f'Saved figures to {args.images_dir}; tables to {args.output_dir}')


if __name__ == '__main__':
    main()
