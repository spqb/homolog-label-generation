#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run the full DELIGHT pipeline (splits -> embeddings -> predictions -> RBM datasets -> RBM training).

Usage:
  src/scripts/run_delight.sh \
    --source_csv <path> \
    --seed <int> \
    --dataset_name <name> \
    --t1 <float> \
    --rbm_model_path <path>

Arguments:
  --source_csv     Path to the CSV file to be used for running the experiment.
  --seed           Random seed.
  --dataset_name   Dataset name used to generate output folders.
  --t1             T1 parameter used for splitting.
  --rbm_model_path Path to the RBM embedding model parameters.
  -h, --help       Show this help message.
EOF
}

SOURCE_CSV=""
SEED=""
DATASET_NAME=""
T1=""
RBM_MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source_csv)
            SOURCE_CSV="$2"
            shift 2
            ;;
        --seed)
            SEED="$2"
            shift 2
            ;;
        --dataset_name)
            DATASET_NAME="$2"
            shift 2
            ;;
        --t1)
            T1="$2"
            shift 2
            ;;
        --rbm_model_path)
            RBM_MODEL_PATH="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$SOURCE_CSV" || -z "$SEED" || -z "$DATASET_NAME" || -z "$T1" || -z "$RBM_MODEL_PATH" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ ! -f "$SOURCE_CSV" ]]; then
    echo "source_csv does not exist: $SOURCE_CSV" >&2
    exit 1
fi

if [[ ! -f "$RBM_MODEL_PATH" ]]; then
    echo "rbm_model_path does not exist: $RBM_MODEL_PATH" >&2
    exit 1
fi

BASE_OUT="outputs/${DATASET_NAME}/t1_${T1}/seed_${SEED}"
SPLITS_DIR="${BASE_OUT}/splits"
EMBEDDINGS_DIR="${BASE_OUT}/embeddings"
PREDICTIONS_DIR="${BASE_OUT}/predictions"
DATASETS_DIR="${BASE_OUT}/datasets_for_rbm"
MODEL_DIR="models/${DATASET_NAME}/t1_${T1}/seed_${SEED}"

# Part 1) Train/test splits
python src/split_train_test.py \
    --source_csv "$SOURCE_CSV" \
    --seed "$SEED" \
    --t1 "$T1" \
    --output_dir "$SPLITS_DIR"

# Part 2) Embeddings creation
src/scripts/create_embeddings.sh \
    --input_folder "$SPLITS_DIR" \
    --embedding_model onehot \
    --output_folder "$EMBEDDINGS_DIR" \
    --column_sequences sequence_align

src/scripts/create_embeddings.sh \
    --input_folder "$SPLITS_DIR" \
    --embedding_model rbm \
    --output_folder "$EMBEDDINGS_DIR" \
    --checkpoint "$RBM_MODEL_PATH" \
    --column_sequences sequence_align

src/scripts/create_embeddings.sh \
    --input_folder "$SPLITS_DIR" \
    --embedding_model plm \
    --output_folder "$EMBEDDINGS_DIR" \
    --column_sequences sequence \
    --bf16

# Part 3) Label inference
src/scripts/run_predictions_from_embeddings.sh \
    --input_folder "$EMBEDDINGS_DIR" \
    --output_folder "$PREDICTIONS_DIR"

# Part 4) Create datasets for conditioned RBM training
python src/prepare_rbm_dataset.py \
    --input_dir "$PREDICTIONS_DIR" \
    --csv_pool "$SOURCE_CSV" \
    --output_dir "$DATASETS_DIR"

# Part 5) Conditioned RBM trainings
src/scripts/automate_rbm_training.sh \
    --input_dir "$DATASETS_DIR" \
    --output_dir "$MODEL_DIR" \
    --nepochs 50000 \
    --hidden 500 \
    --no_reweighting \
    --lr 0.005 \

