# From Scarce Functional Labels to Label-Aware Generation in Homologous Protein Families

This repository contains the code accompanying the study on light-supervision for protein-family annotation and conditional sequence generation. The project focuses on a practical setting where a large homologous family contains only a small number of functionally labeled sequences. The central questions are:

- how well different sequence representations support fine-grained label prediction under scarce supervision;
- whether inferred labels can be propagated to the rest of the family and then used to train label-aware generative models.

The pipeline compares one-hot encodings, family-specific RBM embeddings, and ESM2-based protein language model embeddings for specificity prediction, and then uses the resulting annotations to train conditioned RBMs for label-aware sequence generation.

## Repository contents

- `src/encoding/`: sequence embedding scripts for one-hot, RBM, and PLM representations
- `src/plm/`: supervised fine-tuning utilities for ESM2-based models
- `src/pipeline/`: end-to-end pipeline entrypoints and stage scripts
- `src/prediction/`: prediction utilities for generated samples
- `data/`: family-specific datasets used by the pipeline
- `models/`: trained embedding and generative model checkpoints
- `outputs/`: splits, embeddings, predictions, and generated-sample outputs
- `docs/run_experiments_script_report.md`: detailed documentation of the pipeline structure

## Pipeline overview

The implemented workflow has two main parts.

1. **Annotation with scarce labels**
   - split labeled data into train and test sets while limiting phylogenetic leakage;
   - build sequence embeddings;
   - train predictors on small labeled subsets;
   - evaluate specificity prediction on held-out sequences.

2. **Label-aware generation**
   - use predicted labels to assemble RBM training datasets;
   - train conditioned RBMs on sequences and labels;
   - sample artificial homologs from the conditioned models;
   - re-predict labels on generated sequences for consistency checks.

For the `RR` dataset, the pipeline also includes supervised ESM2 fine-tuning when enough labeled training examples are available.

## Main entrypoints

- `src/pipeline/run_experiments.sh`: run all configured experiments
- `src/pipeline/run_case.sh`: run a single dataset / seed / threshold case
- `src/pipeline/run_experiments_smoke.sh`: run a small isolated smoke test into `tmp/smoke_outputs` and `tmp/smoke_models`

## Installation

The project was developed in a Conda environment. A reproducible environment
specification is provided in [environment.yml](environment.yml).

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate delight
```

After the data archive becomes available, create the `data/` directory from the
Zenodo package before running the pipeline.

## Running the pipeline

Run all configured experiments:

```bash
src/pipeline/run_experiments.sh
```

Run a single isolated smoke test without touching the main `outputs/` and `models/` trees:

```bash
src/pipeline/run_experiments_smoke.sh
```

The smoke configuration currently targets one `RR` case and is intended only to validate that the full pipeline executes coherently.

## Analysis and plots

The figures explored in the original notebooks can also be generated through the scripts in `analysis/`. These scripts provide cleaner command-line entrypoints for the most common plot families:

- `analysis/analyze_splits.py`: train/test split composition, sequence-identity histograms, and PCA projections
- `analysis/analyze_classification.py`: repeated classification summaries, macro-F1 plots, and ROC curves across seeds
- `analysis/analyze_generated_samples.py`: conditioned-generation comparisons, symmetric-KL summaries, self-consistency plots, and generated-sample quality analyses

Typical usage examples:

```bash
python analysis/analyze_splits.py --dataset RR --t1 0.4 --seed 1 --use-latex

python analysis/analyze_classification.py --dataset RR --t1 0.4 --use-latex

python analysis/analyze_generated_samples.py --dataset RR --t1 0.4 --use-latex
```

All three scripts write their outputs by default under:

```text
images/analysis/
```

The scripts assume that the corresponding pipeline outputs already exist under `outputs/` and `models/`.

## Notes

- The repository does **not** include the `models/` and `outputs/` folders used in the experiments, because these artifacts are too large to distribute through the repository.
- The `data/` folder is expected to be created from an external Zenodo archive. The download link will be added once it is available.
- The conditioned-generation branch is active only for the `RR` dataset in the current pipeline configuration.
- The code expects the external tools and Python dependencies required by the embedding, classification, and RBM stages to be available in the execution environment.
- The repository is intended to contain the code and documentation needed to reproduce the experiments once the external data and model artifacts are available.

## Citation

If you use this repository, please cite the associated article:

**Lorenzo Rosset, Martin Weigt, and Francesco Zamponi.**  
*From Scarce Functional Labels to Label-Aware Generation in Homologous Protein Families.*
