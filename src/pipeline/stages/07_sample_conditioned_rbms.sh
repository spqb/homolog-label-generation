#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/07_sample_conditioned_rbms.sh \
    --datasets_dir <path> \
    --model_dir <path> \
    [--annadca_bin annadca]

Samples each trained RBM_labels_* model using its matching RBM dataset CSV.
EOF
}

DATASETS_DIR=""
MODEL_DIR=""
ANNADCA_BIN="annadca"
PARAMS_FILE="params.h5"
SAMPLES_FILE="samples.csv"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets_dir) DATASETS_DIR="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --annadca_bin) ANNADCA_BIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$DATASETS_DIR" || -z "$MODEL_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if ! command -v "$ANNADCA_BIN" >/dev/null 2>&1; then
    echo "annadca executable not found: $ANNADCA_BIN" >&2
    exit 1
fi

if [[ ! -d "$DATASETS_DIR" ]]; then
    echo "datasets_dir does not exist: $DATASETS_DIR" >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "model_dir does not exist: $MODEL_DIR" >&2
    exit 1
fi

shopt -s nullglob
dataset_files=("$DATASETS_DIR"/rbm_dataset_ntrain_*.csv)
model_dirs=("$MODEL_DIR"/RBM_labels_*)
shopt -u nullglob

if [[ ${#dataset_files[@]} -eq 0 ]]; then
    echo "No rbm_dataset_ntrain_*.csv files found in: $DATASETS_DIR" >&2
    exit 1
fi

if [[ ${#model_dirs[@]} -eq 0 ]]; then
    echo "No RBM_labels_* model directories found in: $MODEL_DIR" >&2
    exit 1
fi

largest_dataset="$(
    printf '%s\n' "${dataset_files[@]}" | sort -V | tail -n 1
)"

num_runs=0
for model_path in "${model_dirs[@]}"; do
    [[ -d "$model_path" ]] || continue

    model_name="$(basename "$model_path")"
    dataset_path=""
    label_column=""

    if [[ "$model_name" == "RBM_labels_all_true" ]]; then
        dataset_path="$largest_dataset"
        label_column="true_label"
    elif [[ "$model_name" =~ ^RBM_labels_([0-9]+)_(.+)$ ]]; then
        ntrain="${BASH_REMATCH[1]}"
        label_suffix="${BASH_REMATCH[2]}"
        dataset_path="$DATASETS_DIR/rbm_dataset_ntrain_${ntrain}.csv"
        label_column="${label_suffix}_label"
    else
        echo "Skipping unexpected model directory name: $model_path" >&2
        continue
    fi

    params_path="$model_path/$PARAMS_FILE"
    samples_path="$model_path/$SAMPLES_FILE"

    if [[ ! -f "$dataset_path" ]]; then
        echo "Skipping $model_name: missing dataset $dataset_path" >&2
        continue
    fi

    if [[ ! -f "$params_path" ]]; then
        echo "Skipping $model_name: missing params $params_path" >&2
        continue
    fi

    echo "Sampling $model_name"
    echo "  dataset: $dataset_path"
    echo "  params : $params_path"
    echo "  output : $samples_path"

    "$ANNADCA_BIN" sample \
        -d "$dataset_path" \
        -p "$params_path" \
        -o "$samples_path" \
        --column_name header \
        --column_label "$label_column" \
        --column_sequence sequence_align

    num_runs=$((num_runs + 1))
done

if [[ $num_runs -eq 0 ]]; then
    echo "No RBM sampling runs were executed." >&2
    exit 1
fi

echo "Completed $num_runs RBM sampling runs."
