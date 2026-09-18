# Missing annotations: zero vector versus posterior marginalization

This RR experiment compares two matched annaDCA trainings on the same
train/test union. The zero arm uses the default treatment of missing labels;
the posterior arm adds `--marginalize_missing_labels`. See
[comparison_methods.md](comparison_methods.md) for the objective and metrics.

From the repository root, preview or run both annotation budgets (100 and
1,000) over paired seeds 1–3:

```bash
uv run --locked bash src/pipeline/run_experiments.sh --profile missing-labels --plan
uv run --locked bash src/pipeline/run_experiments.sh --profile missing-labels
```

For a small pilot, add `--train-sizes 100 --seeds 1`. The default training
settings are 500 hidden units, learning rate 0.005, 10,000 updates, 5,000
persistent chains, 10 training Gibbs steps, no sequence reweighting, and
1,000 generated sequences per label after 1,000 sampling Gibbs steps. Both
arms use the same seed. Both arms use the locked annaDCA 0.2.3 release from
PyPI; the posterior arm enables its `--marginalize_missing_labels` option.

The driver is `src/pipeline/missing_labels.py`. Learned parameters are under
`models/missing_labels/RR/t1_0.4/ntrain_<budget>/seed_<seed>/`; targets,
samples, manifests, and logs are under the parallel `outputs/missing_labels/`
path. Per-budget CSV tables are under `outputs/analysis/missing_labels/`, and
all PDF/PNG/SVG figures are under `outputs/images/missing_labels/`. The
read-only metric and summary modules live in `src/analysis/missing_labels/`.

Higher connected-correlation Pearson recovery and lower symmetric KL in PCA
space indicate better distribution recovery. The three splits give descriptive
paired evidence; inspect the per-label tables alongside the macro-averages.
