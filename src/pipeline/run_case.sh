#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run one DELIGHT pipeline case.

Usage:
  src/pipeline/run_case.sh \
    --source_csv <path> \
    --seed <int> \
    --dataset <name> \
    --t1 <float> \
    --rbm_model_path <path> \
    [--outputs_root <path>] \
    [--models_root <path>] \
    [--supervised_plm_enabled 0|1] \
    [--supervised_plm_min_ntrain 1000]

Compatibility aliases:
  --dataset_name is accepted for --dataset.
  --rbm_embedding_model is accepted for --rbm_model_path.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

SOURCE_CSV=""
SEED=""
DATASET_NAME=""
T1=""
RBM_MODEL_PATH=""
OUTPUTS_ROOT="outputs"
MODELS_ROOT="models"
SUPERVISED_PLM_ENABLED=""
SUPERVISED_PLM_MIN_NTRAIN=1000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source_csv) SOURCE_CSV="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --dataset|--dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --t1) T1="$2"; shift 2 ;;
        --rbm_model_path|--rbm_embedding_model) RBM_MODEL_PATH="$2"; shift 2 ;;
        --outputs_root) OUTPUTS_ROOT="$2"; shift 2 ;;
        --models_root) MODELS_ROOT="$2"; shift 2 ;;
        --supervised_plm_enabled) SUPERVISED_PLM_ENABLED="$2"; shift 2 ;;
        --supervised_plm_min_ntrain) SUPERVISED_PLM_MIN_NTRAIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
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

if [[ -z "$SUPERVISED_PLM_ENABLED" ]]; then
    if [[ "$DATASET_NAME" == "RR" ]]; then
        SUPERVISED_PLM_ENABLED=1
    else
        SUPERVISED_PLM_ENABLED=0
    fi
fi

BASE_OUT="${OUTPUTS_ROOT}/${DATASET_NAME}/t1_${T1}/seed_${SEED}"
SPLITS_DIR="${BASE_OUT}/splits"
EMBEDDINGS_DIR="${BASE_OUT}/embeddings"
PREDICTIONS_DIR="${BASE_OUT}/predictions"
DATASETS_DIR="${BASE_OUT}/datasets_for_rbm"
MODEL_DIR="${MODELS_ROOT}/${DATASET_NAME}/t1_${T1}/seed_${SEED}"

echo "Running pipeline case: dataset=${DATASET_NAME}, t1=${T1}, seed=${SEED}"
echo "  outputs_root: ${OUTPUTS_ROOT}"
echo "  models_root : ${MODELS_ROOT}"

src/pipeline/stages/01_split_dataset.sh \
    --source_csv "$SOURCE_CSV" \
    --seed "$SEED" \
    --t1 "$T1" \
    --output_dir "$SPLITS_DIR"

src/pipeline/stages/02_embed_splits.sh \
    --splits_dir "$SPLITS_DIR" \
    --output_dir "$EMBEDDINGS_DIR" \
    --rbm_model_path "$RBM_MODEL_PATH"

src/pipeline/stages/03_predict_test_embeddings.sh \
    --embeddings_dir "$EMBEDDINGS_DIR" \
    --predictions_dir "$PREDICTIONS_DIR"

src/pipeline/stages/04_train_rr_supervised_plm.sh \
    --dataset "$DATASET_NAME" \
    --splits_dir "$SPLITS_DIR" \
    --predictions_dir "$PREDICTIONS_DIR" \
    --model_dir "$MODEL_DIR" \
    --seed "$SEED" \
    --enabled "$SUPERVISED_PLM_ENABLED" \
    --min_ntrain "$SUPERVISED_PLM_MIN_NTRAIN"

if [[ "$DATASET_NAME" == "RR" ]]; then
    src/pipeline/stages/05_prepare_rbm_datasets.sh \
        --predictions_dir "$PREDICTIONS_DIR" \
        --source_csv "$SOURCE_CSV" \
        --output_dir "$DATASETS_DIR"

    src/pipeline/stages/06_train_conditioned_rbms.sh \
        --datasets_dir "$DATASETS_DIR" \
        --model_dir "$MODEL_DIR"

    src/pipeline/stages/07_sample_conditioned_rbms.sh \
        --datasets_dir "$DATASETS_DIR" \
        --model_dir "$MODEL_DIR"

    src/pipeline/stages/08_predict_generated_samples.sh \
        --source_csv "$SOURCE_CSV" \
        --splits_dir "$SPLITS_DIR" \
        --embeddings_dir "$EMBEDDINGS_DIR" \
        --model_dir "$MODEL_DIR" \
        --rbm_model_path "$RBM_MODEL_PATH"
else
    echo "Skipping conditioned RBM stages for dataset=${DATASET_NAME}"
fi

echo "Completed pipeline case: dataset=${DATASET_NAME}, t1=${T1}, seed=${SEED}"
