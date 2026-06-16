# Pipeline Script Report

This document describes the nested script structure launched by
`src/pipeline/run_experiments.sh` and summarizes what each pipeline stage does.

## Top-Level Entrypoint

### `src/pipeline/run_experiments.sh`

`src/pipeline/run_experiments.sh` is the batch launcher for repeated DELIGHT
runs. It reads dataset configuration files from `src/pipeline/config/` and
calls `src/pipeline/run_case.sh` for every configured seed and `t1` value.

Active experiment grid:

| Dataset | Source CSV | Seeds | `t1` values | RBM checkpoint |
| --- | --- | --- | --- | --- |
| `cm` | `data/cm/cm_tested_seqs.csv` | `1..10` | `0.7` | `models/cm/embedding_rbm.h5` |
| `Globin` | `data/Globin/Globin_morkos.csv` | `1..10` | `0.7` | `models/Globin/embedding_rbm.h5` |
| `SH3` | `data/SH3/SH3.csv` | `1..10` | `0.7` | `models/SH3/embedding_rbm.h5` |
| `RR` | `data/RR/RR.csv` | `1..10` | `0.4`, `0.7` | `models/RR/embedding_rbm_ptt.h5` |

Config files:

```text
src/pipeline/config/cm.conf
src/pipeline/config/Globin.conf
src/pipeline/config/SH3.conf
src/pipeline/config/RR.conf
src/pipeline/config/smoke_rr.conf
```

`src/pipeline/config/smoke_rr.conf` is a one-case RR smoke-test config:

```bash
src/pipeline/run_experiments.sh \
  --outputs_root tmp/smoke_outputs \
  --models_root tmp/smoke_models \
  src/pipeline/config/smoke_rr.conf
```

For each run, `run_experiments.sh` calls:

```bash
src/pipeline/run_case.sh \
  --source_csv <dataset csv> \
  --seed <seed> \
  --dataset <dataset name> \
  --t1 <threshold> \
  --rbm_model_path <checkpoint> \
  --outputs_root <path> \
  --models_root <path> \
  --supervised_plm_enabled <0|1> \
  --supervised_plm_min_ntrain <N>
```

Both orchestrators default to `outputs/` and `models/`, but they now accept
`--outputs_root` and `--models_root` so the pipeline can be smoke-tested in an
isolated scratch location without touching the main artifacts.

## Active Call Tree

```text
src/pipeline/run_experiments.sh
+-- src/pipeline/config/*.conf
+-- src/pipeline/run_case.sh
    +-- src/pipeline/stages/01_split_dataset.sh
    |   +-- python src/split_train_test.py
    +-- src/pipeline/stages/02_embed_splits.sh
    |   +-- python src/encoding/onehot_encoding.py
    |   +-- python src/encoding/rbm_encoding.py
    |   +-- python src/encoding/plm_encoding.py
    +-- src/pipeline/stages/03_predict_test_embeddings.sh
    |   +-- python src/predict_from_embeddings.py
    +-- src/pipeline/stages/04_train_rr_supervised_plm.sh
    |   +-- python src/plm/train_supervised.py
    |   +-- python src/predict_from_plm_supervised_freeze.py
    +-- [RR only] src/pipeline/stages/05_prepare_rbm_datasets.sh
    |   +-- python src/prepare_rbm_dataset.py
    +-- [RR only] src/pipeline/stages/06_train_conditioned_rbms.sh
    |   +-- annadca train
    +-- [RR only] src/pipeline/stages/07_sample_conditioned_rbms.sh
    |   +-- annadca sample
    +-- [RR only] src/pipeline/stages/08_predict_generated_samples.sh
        +-- python src/encoding/onehot_encoding.py
        +-- python src/encoding/rbm_encoding.py
        +-- python src/encoding/plm_encoding.py
        +-- python src/predict_from_embeddings.py
        +-- python src/predict_from_plm_supervised_freeze.py
        +-- python src/prediction/predict_from_csv_true.py
```

## Pipeline Orchestrator

### `src/pipeline/run_case.sh`

`src/pipeline/run_case.sh` runs the full per-dataset/per-seed/per-`t1`
pipeline. It builds the output layout from the dataset name, threshold, and
seed:

```text
outputs/<dataset_name>/t1_<t1>/seed_<seed>/
+-- splits/
+-- embeddings/
+-- predictions/
+-- datasets_for_rbm/   # RR only

models/<dataset_name>/t1_<t1>/seed_<seed>/
```

The case runner accepts optional root overrides:

```bash
--outputs_root <path>
--models_root <path>
```

For example, a scratch RR smoke test can run into isolated folders:

```bash
src/pipeline/run_case.sh \
  --source_csv data/RR/RR.csv \
  --seed 1 \
  --dataset RR \
  --t1 0.4 \
  --rbm_model_path models/RR/embedding_rbm_ptt.h5 \
  --outputs_root tmp/smoke_outputs \
  --models_root tmp/smoke_models
```

The case runner performs these stages:

1. `src/pipeline/stages/01_split_dataset.sh`
2. `src/pipeline/stages/02_embed_splits.sh`
3. `src/pipeline/stages/03_predict_test_embeddings.sh`
4. `src/pipeline/stages/04_train_rr_supervised_plm.sh`
5. `src/pipeline/stages/05_prepare_rbm_datasets.sh` for `RR` only
6. `src/pipeline/stages/06_train_conditioned_rbms.sh` for `RR` only
7. `src/pipeline/stages/07_sample_conditioned_rbms.sh` for `RR` only
8. `src/pipeline/stages/08_predict_generated_samples.sh` for `RR` only

## Stage 1: Split Creation

### `src/split_train_test.py`

This Python script prepares train and test CSV splits from the source dataset.
It expects the source CSV to contain at least `header`, `sequence_align`, and
`label` columns.

Main behavior:

- Drops rows where `label` is missing.
- Processes each label independently.
- Encodes aligned protein sequences with `adabmDCA.fasta.encode_sequence`.
- Uses `adabmDCA.cobalt.run_cobalt` to split sequences by similarity thresholds.
- Repeats Cobalt splitting up to 50 times if a label produces an empty train or
  test set.
- Excludes labels that have too few train or test examples, controlled by
  `--min_data_per_label` with default `10`.
- Writes balanced train subsets for each requested total train size.
- Writes one balanced test set.

Default train subset sizes:

```text
5000, 2000, 1000, 500, 100
```

Outputs written under `outputs/<dataset>/t1_<t1>/seed_<seed>/splits/`:

```text
train_5000.csv
train_2000.csv
train_1000.csv
train_500.csv
train_100.csv
test.csv
```

## Stage 2: Split Embeddings

### `src/pipeline/stages/02_embed_splits.sh`

This stage scans the split folder for:

```text
train_<N>.csv
test.csv
```

For each supported CSV file, it creates an embedding HDF5 file named:

```text
<input_basename>.embedding.<model>.h5
```

It also sets an `info` metadata string:

```text
train-<N>-<model>
test-0-<model>
```

Supported embedding modes:

| Mode | Called Python script | Required extra argument | Sequence column in the pipeline |
| --- | --- | --- | --- |
| `onehot` | `src/encoding/onehot_encoding.py` | none | `sequence_align` |
| `rbm` | `src/encoding/rbm_encoding.py` | `--checkpoint <rbm_model_path>` | `sequence_align` |
| `plm` | `src/encoding/plm_encoding.py` | optional checkpoint | `sequence` |

It creates embeddings for `onehot`, `rbm`, and `plm`. The PLM calls pass
`--bf16`, which enables bfloat16 autocast only when CUDA and GPU support are
available.

## Stage 2a: One-Hot Encoding

### `src/encoding/onehot_encoding.py`

This script one-hot encodes aligned protein sequences.

Main behavior:

- Loads sequences, headers, and labels from a CSV through `utils.load_query_data`.
- Uses the protein token alphabet from `adabmDCA.fasta.get_tokens("protein")`.
- Integer-encodes sequences with `encode_sequence`.
- Converts each residue position to one-hot form.
- Flattens each encoded sequence to shape:

```text
n_sequences x (sequence_length * n_tokens)
```

Output HDF5 datasets:

```text
info
embeddings
headers
labels   # only when labels are present
```

## Stage 2b: RBM Encoding

### `src/encoding/rbm_encoding.py`

This script encodes aligned protein sequences using a trained RBM model.

Main behavior:

- Loads the input CSV with `utils.load_query_data`.
- Loads the latest saved RBM parameters from the supplied model file.
- Encodes aligned sequences into token indices.
- Sends the encoded sequences through the RBM hidden layer sampler.
- Saves the hidden magnetizations, `hidden_mag`, as the embedding matrix.

Output HDF5 datasets:

```text
info
embeddings
headers
labels   # only when labels are present
```

## Stage 2c: PLM Encoding

### `src/encoding/plm_encoding.py`

This script embeds unaligned protein sequences with a Hugging Face protein
language model. By default it uses:

```text
facebook/esm2_t33_650M_UR50D
```

Main behavior:

- Loads a tokenizer and `AutoModel`.
- Optionally loads a fine-tuned backbone from `--checkpoint`.
- Reads sequences, headers, and labels from the input CSV.
- Tokenizes sequences with padding/truncation to `--max_length`, default `256`.
- Runs the model in evaluation mode.
- Removes the CLS token and mean-pools non-padding token embeddings.
- Saves one embedding vector per sequence.

Output HDF5 datasets:

```text
info
embeddings
headers
labels   # only when labels are present
```

## Stage 3: Test Embedding Predictions

### `src/pipeline/stages/03_predict_test_embeddings.sh`

This stage scans the embedding output folder for HDF5 files and pairs compatible
train/test embeddings before calling `src/predict_from_embeddings.py`.

Expected naming patterns:

```text
train_<N>.embedding.<model>.h5
test.embedding.<model>.h5
test.embedding.supervised.<N>.h5
```

Current important behavior:

- It records generic test embeddings by model name.
- It has special handling for `test.embedding.supervised.<N>.h5`.
- It runs predictions for every compatible train/test pair. For `supervised`
  embeddings, compatibility is keyed by both model name and train size; for
  other embeddings, compatibility is keyed by model name.

For each compatible pair, it writes:

```text
outputs/<dataset>/t1_<t1>/seed_<seed>/predictions/
+-- test.embedding.<model>.ntrain_<N>.predictions.h5
```

## Stage 3a: Prediction From Embeddings

### `src/predict_from_embeddings.py`

This script trains conventional classifiers on train embeddings and predicts
labels for test embeddings.

Main behavior:

- Loads `embeddings`, `headers`, and optional `labels` from train/test HDF5
  files.
- Requires labels in the training HDF5 file.
- Standardizes train and test embeddings with `StandardScaler`.
- Trains three classifiers:
  - `LogisticRegression(max_iter=1000)`
  - linear-kernel `SVC(probability=True, random_state=42)`
  - `RandomForestClassifier(n_estimators=100, random_state=42)`
- Predicts labels and probabilities for the test set.
- Reports accuracy when test labels are available.

Output HDF5 structure:

```text
info
train/
  headers
  labels_true
test/
  headers
  labels_true                  # only when test labels are present
  predictions/
    logreg/
      labels_pred
      labels_probs
    SVM/
      labels_pred
      labels_probs
    random_forest/
      labels_pred
      labels_probs
```

## Stage 3b: RR Supervised PLM Finetuning

### `src/plm/train_supervised.py`

For `RR` runs only, `src/pipeline/stages/04_train_rr_supervised_plm.sh` loops
over split files named `train_<N>.csv` and calls this script when `N >= 1000`.

Active train sizes from the default splitter:

```text
1000, 2000, 5000
```

Main behavior:

- Loads `train_<N>.csv` as the supervised training set.
- Uses `test.csv` as an optional evaluation set.
- Fine-tunes an ESM2 sequence classifier with the default backbone:

```text
facebook/esm2_t33_650M_UR50D
```

- Saves the fine-tuned backbone, tokenizer, classifier head, metrics log, and
  label mapping.
- Stores both `label2id` and `id2label` so prediction outputs can use the
  original label names.

Output directory:

```text
models/RR/t1_<t1>/seed_<seed>/pLM_encoder_<N>_supervised/
```

The active call in `src/pipeline/stages/04_train_rr_supervised_plm.sh` uses:

```bash
python src/plm/train_supervised.py \
  --train_csv "$SPLITS_DIR/train_<N>.csv" \
  --test_csv "$SPLITS_DIR/test.csv" \
  --folder_params "$MODEL_DIR/pLM_encoder_<N>_supervised" \
  --epochs 50 \
  --seed "$SEED" \
  --bf16
```

## Stage 3c: RR Supervised PLM Prediction

### `src/predict_from_plm_supervised_freeze.py`

After each RR supervised PLM model is trained,
`src/pipeline/stages/04_train_rr_supervised_plm.sh` calls this script to predict
labels for `test.csv`.

Main behavior:

- Loads the fine-tuned backbone from `--model_dir`.
- Loads the saved `classifier_head.pt`.
- Loads label mappings from `label_mapping.json`.
- Predicts labels and probabilities for the test split.
- When `--train_csv` is provided, writes a `train` group containing training
  headers and true labels. This makes the output compatible with
  `src/prepare_rbm_dataset.py`.

Output file:

```text
outputs/RR/t1_<t1>/seed_<seed>/predictions/
+-- test.embedding.plm_supervised.ntrain_<N>.predictions.h5
```

Output HDF5 structure:

```text
info                         # train-<N>-plm_supervised
train/
  headers
  labels_true
test/
  headers
  labels_true
  predictions/
    plm_supervised/
      labels_pred
      labels_probs
```

## Stage 5: RBM Dataset Preparation

### `src/prepare_rbm_dataset.py`

For `RR` runs only, this script merges prediction HDF5 files into CSV datasets
for conditioned RBM training.

Main behavior:

- Recursively scans `--input_dir` for prediction `.h5` files.
- Reads each file's `info` field, expected to follow:

```text
<set>-<n_train>-<model>
```

- Uses `--csv_pool` to map sequence headers back to aligned sequences from the
  original source CSV.
- Groups rows by `n_train`.
- For train rows, copies the true label into each model-specific label column.
- For test rows, fills the model-specific label column with predicted labels
  from one classifier.
- Chooses the classifier named by `--classifier`, default `logreg`; if absent,
  it falls back to `random_forest`, then `logreg`, then `SVM`, then the first
  available group.

Outputs written under
`outputs/<dataset>/t1_<t1>/seed_<seed>/datasets_for_rbm/`:

```text
rbm_dataset_ntrain_<N>.csv
```

Each output CSV has columns:

```text
header
sequence_align
set
true_label
<model>_label ...
train_only
```

## Stage 6: Conditioned RBM Training

### `src/pipeline/stages/06_train_conditioned_rbms.sh`

For `RR` runs only, this stage runs after RBM dataset preparation. It
automates `annadca train` runs over the CSV files produced by
`src/prepare_rbm_dataset.py`.

Main behavior:

- Recursively finds:

```text
rbm_dataset_ntrain_*.csv
```

- Detects label columns automatically as columns ending in `_label`, and also
  includes `train_only`.
- Trains one model per detected label column and train size.
- Treats `true_label` specially: it trains `RBM_labels_all_true` once, rather
  than once per train size.
- For other label columns, writes model directories named:

```text
RBM_labels_<ntrain>_<label_prefix>
```

- Supports `--dry_run` to print planned commands without executing them.

The stage uses these fixed training settings:

```text
--nepochs 50000
--hidden 500
--no_reweighting
--lr 0.005
```

Each generated `annadca train` command includes:

```bash
annadca train \
  -d <rbm_dataset_csv> \
  -o <model_dir> \
  --column_sequences sequence_align \
  --column_labels <label_column> \
  --column_name header
```

## Stage 7: RBM Sampling

### `src/pipeline/stages/07_sample_conditioned_rbms.sh`

For `RR` runs only, this stage runs after conditioned RBM training. It samples
every trained `RBM_labels_*` model directory under the current `MODEL_DIR`.

Main behavior:

- Reads RBM datasets from:

```text
outputs/<dataset>/t1_<t1>/seed_<seed>/datasets_for_rbm/
```

- Reads trained RBM model folders from:

```text
models/<dataset>/t1_<t1>/seed_<seed>/
```

- Matches model folders named `RBM_labels_<N>_<label>` to
  `rbm_dataset_ntrain_<N>.csv`.
- Samples `RBM_labels_all_true`, when present, with the largest available
  `rbm_dataset_ntrain_<N>.csv`.
- Writes `samples.csv` inside each sampled model directory.

Each generated `annadca sample` command includes:

```bash
annadca sample \
  -d <rbm_dataset_csv> \
  -p <model_dir>/params.h5 \
  -o <model_dir>/samples.csv \
  --column_name header \
  --column_label <label_column> \
  --column_sequence sequence_align
```

## Stage 8: Generated-Sample Predictions

### `src/pipeline/stages/08_predict_generated_samples.sh`

For `RR` runs only, this stage runs after RBM sampling. It embeds generated
samples and writes generated-sample prediction files.

Main behavior:

- Iterates over `RBM_labels_*` folders in the current `MODEL_DIR`.
- Reads each folder's `samples.csv`.
- Creates the required `samples.embedding.<model>.h5` file for generated
  samples when an embedding-based classifier is used.
- Writes `samples.predictions.<model>.h5` beside each `samples.csv`.

Model-specific behavior:

- `RBM_labels_all_true`: one-hot encodes samples as
  `samples.embedding.true.h5`, then calls
  `src/prediction/predict_from_csv_true.py` in single-folder mode.
- `RBM_labels_<N>_onehot`: one-hot encodes samples, then calls
  `src/predict_from_embeddings.py` with
  `train_<N>.embedding.onehot.h5`.
- `RBM_labels_<N>_rbm`: RBM-encodes samples with the original RBM embedding
  checkpoint, then calls `src/predict_from_embeddings.py` with
  `train_<N>.embedding.rbm.h5`.
- `RBM_labels_<N>_plm`: PLM-encodes samples, then calls
  `src/predict_from_embeddings.py` with `train_<N>.embedding.plm.h5`.
- `RBM_labels_<N>_plm_supervised`: calls
  `src/predict_from_plm_supervised_freeze.py` using the matching
  `pLM_encoder_<N>_supervised` model.

### `src/prediction/predict_from_csv_true.py`

This script is the helper used for the `RBM_labels_all_true` generated-sample
case. It supports two modes:

- `--query_source <path>` for the previous batch behavior over `seed_*`
  directories.
- `--query_dir <path>` for the current pipeline's single
  `RBM_labels_all_true` folder.

In pipeline mode, `src/pipeline/stages/08_predict_generated_samples.sh` first creates
`samples.embedding.true.h5`, then this script trains a logistic regression on
the original source CSV using one-hot encoded aligned sequences and writes:

```text
<model_dir>/RBM_labels_all_true/samples.predictions.true.h5
```
