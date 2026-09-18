# From Scarce Functional Labels to Label-Aware Generation in Homologous Protein Families

This repository contains the code accompanying the study on scarce-label protein-family annotation and conditional sequence generation. It evaluates specificity prediction for RR, Globin, SH3, and cm, then evaluates conditional generation for RR. The study asks:

- how well different sequence representations support fine-grained label prediction under scarce supervision;
- whether inferred labels can be propagated to the rest of the family and then used to train label-aware generative models.

Classification compares one-hot encodings, family-specific RBM embeddings, and frozen ESM-2, ESM-C, and Ankh protein language model embeddings. RR also has a supervised ESM-2 experiment for budgets of at least 1,000 annotated sequences. The RR generation experiment trains conditioned RBMs using ground-truth or predicted annotations.

## Repository contents

- `src/encoding/`: sequence embedding scripts for one-hot, RBM, and PLM representations
- `src/plm/`: RR supervised ESM-2 training utilities
- `src/pipeline/`: experiment configuration, entry point, and stage scripts
- `src/analysis/`: metrics and figures
- `data/`: family source CSVs (local, not tracked by Git)
- `models/`: embedding checkpoints and trained models (local, not tracked by Git)
- `outputs/`: splits, embeddings, predictions, samples, analysis tables, figures, and run logs
- `docs/run_experiments_script_report.md`: profiles, stages, artifact reuse, and output layout

## Pipeline overview

The configured workflow has two main parts.

1. **Annotation with scarce labels**
   - use Cobalt to make train/test splits for each family and seed;
   - create nominal training budgets of 100, 500, 1,000, 2,000, and 5,000 sequences;
   - embed the splits, fit classifiers, and evaluate predictions on the held-out set.

2. **Label-aware generation**
   - for RR, combine the train and test sequences with true, train-only, one-hot, RBM, and Ankh label columns;
   - train conditioned RBMs on these unions and sample from random initial chains;
   - compare generated sequences with natural sequences and evaluate self-consistency.

## Entry point

`src/pipeline/run_experiments.sh` is the public entry point. It reads
`src/pipeline/experiments.toml` and runs the selected profile. The scripts under
`src/pipeline/stages/` are its internal steps; stage 9 invokes `src/analysis/`.

## Installation

Dependencies are managed with [uv](https://docs.astral.sh/uv/). Install uv
using the [official installation instructions](https://docs.astral.sh/uv/getting-started/installation/)
and ensure Git is available (ESM C and Transformers use pinned Git revisions).

The lockfile installs annaDCA 0.2.3 from PyPI, including posterior treatment
of missing labels and the training and sampling commands.

From the repository root, install Python 3.12 and the locked dependencies:

```bash
uv python install 3.12
uv sync --locked
```

`uv sync` creates a project-local `.venv`. Direct dependencies are declared in
[pyproject.toml](pyproject.toml), and [uv.lock](uv.lock) pins the resolved dependency
graph, including the published annaDCA release.
[.python-version](.python-version) selects Python 3.12. The setup includes
PyTorch; GPU inference requires a compatible NVIDIA driver. Protein language
model checkpoints are downloaded on first use, while the family RBM embedding
checkpoints are supplied by the data archive.
Plotting with `--use-latex` also requires a system LaTeX installation; omit that
flag to use Matplotlib's built-in text rendering.

Run commands with `uv run --locked` as shown below. This also puts the project's
Python and command-line tools on `PATH` for nested pipeline scripts. Alternatively,
activate `.venv` with `source .venv/bin/activate` and invoke scripts directly.
In your IDE, select `.venv/bin/python` as the project interpreter.

To change dependencies, edit `pyproject.toml`, run `uv lock`, then `uv sync --locked`;
include the updated lockfile with the change. Use `uv add` to add a new dependency.

Verify the installation with the test suite:

```bash
uv run --locked python -m unittest discover -s tests -v
```

The tests include a small CPU training-and-sampling run through stages 6 and 7
and a posterior missing-label training run. They require neither downloaded
protein-model weights nor experiment data.

Download the data and RBM embedding checkpoints from Zenodo:

```bash
wget https://zenodo.org/records/20719564/files/Data_delight.zip
```

This downloads a zip archive named `Data_delight.zip`. Once the download has
finished, create the local `data/` and `models/` directories with:

```bash
./setup_data.sh Data_delight.zip
```

The setup script requires `unzip` and extracts datasets into `data/` and
embedding RBM checkpoints into `models/`. It refuses to overwrite existing
directories unless called with `--force`. Before running the configured matrix,
check that the four source CSVs and embedding checkpoints named in
`src/pipeline/experiments.toml` are present.
In particular, cm uses `data/cm/cm_sequences_merged.csv`; supply that merged
source separately if it is not present after extraction.

## Running the pipeline

Preview the work before launching expensive stages:

```bash
uv run --locked bash src/pipeline/run_experiments.sh --profile main --plan
```

Run the configured matrix (classification for all four families; generation
for RR):

```bash
uv run --locked bash src/pipeline/run_experiments.sh --profile main
```

Useful narrower selections:

```bash
# Build or reuse cm Cobalt splits, embeddings, predictions, and analysis.
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile classification --datasets cm --t1 0.7

# Run RR generation and its analysis, reusing completed artifacts.
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile generation --datasets RR

# Render the RR PCA comparison from existing unions and samples (seed 1).
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile generation-pca --t1 0.4 --seeds 1

# Compare zero-vector and posterior treatment of missing annotations.
uv run --locked bash src/pipeline/run_experiments.sh \
  --profile missing-labels --datasets RR --t1 0.4
```

Use `--datasets`, `--t1`, `--seeds`, `--train-sizes`, `--methods`,
`--generation-methods`, and `--stages` to narrow work. `--force` refreshes
selected stages; `--plan` previews actions without writing artifacts. The
runner checks completion records and input signatures to reuse valid artifacts.
See [the runner report](docs/run_experiments_script_report.md) for profiles,
prerequisites, stage behavior, and paths.

The quantitative RR generation analysis uses True, one-hot, RBM, and Ankh
conditioned models. Train-only is trained and sampled for seed 1, but appears
only in the RR PCA comparison. Natural-versus-generated sampling always
prescribes `true_label` and starts random chains. Self-consistency instead
prescribes 3,000 targets per model label and uses the corresponding annotation
predictor on the generated sequences. RR sampling uses 1,000 Gibbs steps.

The cm source is `data/cm/cm_sequences_merged.csv`. If it changes, rerun the cm
classification profile from stage 1; analysis alone cannot verify or rebuild
the dependent artifacts. Its embedding RBM checkpoint is used as a fixed input.

## Code and outputs

- `src/pipeline/`: experiment configuration, orchestration, and stages.
- `src/analysis/`: metric computation and figure rendering.
- `models/`: learned parameters and pretrained checkpoints.
- `outputs/<dataset>/t1_<t1>/seed_<seed>/`: splits, embeddings, predictions,
  and a case manifest; RR also has RBM unions and `generation/` samples.
- `outputs/analysis/`: classification, natural-generation,
  self-consistency, and missing-label metric tables.
- `outputs/images/`: figures; this is the only part of `outputs/` eligible for
  Git tracking under the repository's `.gitignore` rules.
- `outputs/logs/`: per-run logs and the resolved run selection.
- `models/<dataset>/t1_<t1>/seed_<seed>/`: conditioned RBM parameters;
  missing-label comparison models are under `models/missing_labels/`.

## Citation

If you use this repository, please cite the associated article:

**Lorenzo Rosset, Martin Weigt, and Francesco Zamponi.**  
*From Scarce Functional Labels to Label-Aware Generation in Homologous Protein Families.*
