# Experiment runner

Run experiments from the repository root with `src/pipeline/run_experiments.sh`.
The shell entry point calls `src/pipeline/run_experiments.py`, which reads the
dataset matrix and defaults from `src/pipeline/experiments.toml`. The nine stage
scripts live in `src/pipeline/stages/`; stage 9 invokes modules in
`src/analysis/`. Use `uv run --locked` to make the pinned Python environment
and annaDCA executable available to every stage.

```bash
uv run --locked bash src/pipeline/run_experiments.sh --profile main --plan
uv run --locked bash src/pipeline/run_experiments.sh --profile main
```

`--plan` prints an artifact inventory without running stages or writing output.
Actual runs write a log, resolved configuration, plan table, and completion
status under `outputs/logs/<run_id>/`. The runner processes one case at a time;
`--jobs` currently accepts only `1`.

## Profiles and selection

| Profile | Configured work |
| --- | --- |
| `main` | Classification and split diagnostics for RR, Globin, SH3, and cm; generation and its analysis for RR only. |
| `classification` | Stages 1–4 and classification analysis for selected families. Stage 4 applies to RR only. |
| `annotation-refresh` | Check or build embeddings and test predictions, then run classification analysis (stages 2, 3, 9). Use `--force` to rebuild completed artifacts. |
| `generation` | RR RBM datasets, training, sampling, generated-sample prediction, and generation analysis (stages 5–9). |
| `generation-pca` | Render the RR PCA comparison from existing RBM unions and samples (stage 9; seed 1 by default). |
| `missing-labels` | Matched RR zero-vector and posterior-marginalization experiments and their analysis. |
| `smoke` | One RR case at t1 = 0.4 with stages 1–3 and 9. This still runs the configured encoders. |

RR is configured for t1 = 0.4 and 0.7; Globin, SH3, and cm use t1 = 0.7.
Each dataset has seeds 1–10. The nominal annotation budgets are 100, 500,
1,000, 2,000, and 5,000. The split procedure balances labels and may return
fewer than the nominal count when a class has too few training sequences;
`manifest.json` in each case records the actual counts.

Use `--datasets`, `--t1`, and `--seeds` to select cases. `--train-sizes`
selects among the configured split budgets when those files already exist.
`--methods` selects classification representations; `--generation-methods`
selects RR generators. `--stages` accepts numbers, ranges, or stage names
(`split`, `embed`, `predict`, `supervised`, `rbm-dataset`, `train`, `sample`,
`generated-predict`, `analyze`). `--force` accepts a stage number or name to
refresh it. `--outputs-root`, `--models-root`, `--annadca-bin`, and
`--use-latex` override the corresponding defaults.

Examples:

```bash
# Build or reuse the cm classification chain from its configured source.
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile classification --datasets cm --t1 0.7

# Run RR generation at both configured t1 values.
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile generation --datasets RR

# Regenerate only the seed-1 RR PCA figure from existing unions and samples.
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile generation-pca --t1 0.4 --seeds 1

# Analyze existing classification artifacts without retraining.
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile classification --datasets RR --stages 9
```

Stage 9 requires its upstream files. For cm, it also requires completion
records matching `data/cm/cm_sequences_merged.csv` for the splits, embeddings,
and test predictions. If those records are missing or stale, run the full cm
`classification` profile before selecting stage 9 alone. The configured cm
embedding RBM is a fixed input to classification.

## Stage contract

| Stage | Input and output |
| --- | --- |
| 1 — split | Cobalt creates one test set and five balanced training subsets from the configured source CSV. |
| 2 — embed | One-hot, family RBM, frozen ESM-2, ESM-C, and Ankh encode train/test splits. |
| 3 — predict | Classifiers fitted to each training embedding predict held-out labels. |
| 4 — supervised | RR supervised ESM-2 trains and predicts for budgets of at least 1,000. |
| 5 — RBM dataset | RR train/test sequences are combined into one CSV per budget with `true_label`, `train_only`, and one-hot, RBM, and Ankh annotation columns. |
| 6 — train | RR conditioned RBMs train on those unions for 50,000 parameter updates. The True model uses the 5,000-budget union. |
| 7 — sample | RR RBMs generate separate natural-comparison and self-consistency samples using 1,000 Gibbs steps. |
| 8 — generated prediction | The predictor corresponding to each RR generator predicts its self-consistency samples. |
| 9 — analyze | Classification, split, RR generation, and missing-label analyses write tables and figures. |

Stages 5–8 and quantitative generation analysis are configured for RR only.
The quantitative generators are True, one-hot, RBM, and Ankh. Seed 1 also
trains and samples train-only models for the RR PCA comparison; train-only is
excluded from the quantitative natural-comparison and self-consistency metrics.

Each stage-6 model is trained on the union of its training and test sequences.
For the natural comparison, stage 7 prescribes each row's `true_label` for
every model, so the generated label counts match its natural union. For
self-consistency, it creates 3,000 targets per model label, samples independent
randomly initialized chains, and stage 8 re-predicts the labels with the
predictor that supplied the generator's annotations. The True benchmark uses
a classifier trained on the full true-label union. Train-only has no
self-consistency predictor.

The `missing-labels` profile is separate from the main generation matrix. It
compares the default zero-vector treatment with annaDCA 0.2.3 posterior
marginalization at budgets 100 and 1,000 over seeds 1–3, using t1 = 0.4 by
default. It requires the corresponding RR stage-5 union CSVs. See
[the missing-label notes](missing_labels/README.md).

## Artifacts and reuse

The active source and fixed embedding checkpoint for each family are specified
in `src/pipeline/experiments.toml`:

| Family | Source CSV | Embedding RBM |
| --- | --- | --- |
| RR | `data/RR/RR.csv` | `models/RR/embedding_rbm_ptt.h5` |
| Globin | `data/Globin/Globin_morkos.csv` | `models/Globin/embedding_rbm.h5` |
| SH3 | `data/SH3/SH3.csv` | `models/SH3/embedding_rbm.h5` |
| cm | `data/cm/cm_sequences_merged.csv` | `models/cm/embedding_rbm.h5` |

Learned model parameters are stored under
`models/<dataset>/t1_<t1>/seed_<seed>/`. Splits, embeddings, test predictions,
RBM unions, samples, and generated-sequence predictions are stored under
`outputs/<dataset>/t1_<t1>/seed_<seed>/`. Missing-label models and run data
use `models/missing_labels/` and `outputs/missing_labels/`.

Analysis tables are written to `outputs/analysis/classification/`,
`outputs/analysis/generation_natural/`,
`outputs/analysis/generation_self_consistency/`, and
`outputs/analysis/missing_labels/`. Figures follow the same grouping under
`outputs/images/`; split figures are under `outputs/images/splits/`. RR PCA
output includes a four-column comparison and a separate True-label panel.

For supported stages, `.pipeline.json` completion records contain input
signatures and output checksums. The runner reuses matching artifacts, reports
missing prerequisites as `BLOCKED`, and rebuilds stale artifacts. It writes
rebuilt file and model artifacts to temporary locations before replacing
completed results. Analysis runs on demand when stage 9 is selected. Review
the `--plan` output before an expensive run, especially after changing a
source CSV, model checkpoint, script, or selected budget.
