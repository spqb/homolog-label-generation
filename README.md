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

## Installation

The project was developed in a Conda environment. A reproducible environment
specification is provided in [environment.yml](environment.yml).

Create and activate the environment with:

```bash
conda env create -f environment.yml
conda activate delight
```

Download the data and RBM embedding checkpoints from Zenodo:

```bash
wget https://zenodo.org/records/20719564/files/Data_delight.zip
```

This downloads a zip archive named `Data_delight.zip`. Once the download has
finished, create the local `data/` and `models/` directories with:

```bash
./setup_data.sh Data_delight.zip
```

The setup script extracts the archive in a temporary directory, copies the
datasets into `data/`, copies the provided RBM checkpoints into `models/`, and
removes the temporary files when it finishes. Existing `data/` or `models/`
directories are not overwritten unless the script is run with `--force`.

## Running the pipeline

Run all configured experiments (this operation will take a lot of time!):

```bash
src/pipeline/run_experiments.sh
```

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
- The `data/` and `models/` folders are expected to be created from the external Zenodo archive with `setup_data.sh`.
- The conditioned-generation branch is active only for the `RR` dataset in the current pipeline configuration.
- The code expects the external tools and Python dependencies required by the embedding, classification, and RBM stages to be available in the execution environment.
- The repository is intended to contain the code and documentation needed to reproduce the experiments once the external data and model artifacts are available.

## Citation

If you use this repository, please cite the associated article:

**Lorenzo Rosset, Martin Weigt, and Francesco Zamponi.**  
*From Scarce Functional Labels to Label-Aware Generation in Homologous Protein Families.*
