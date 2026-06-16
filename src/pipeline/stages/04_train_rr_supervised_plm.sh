#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/04_train_rr_supervised_plm.sh \
    --dataset <name> \
    --splits_dir <path> \
    --predictions_dir <path> \
    --model_dir <path> \
    --seed <int> \
    [--enabled 0|1] \
    [--min_ntrain 1000]
EOF
}

DATASET_NAME=""
SPLITS_DIR=""
PREDICTIONS_DIR=""
MODEL_DIR=""
SEED=""
ENABLED=0
MIN_NTRAIN=1000

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dataset|--dataset_name) DATASET_NAME="$2"; shift 2 ;;
        --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
        --predictions_dir) PREDICTIONS_DIR="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --enabled) ENABLED="$2"; shift 2 ;;
        --min_ntrain) MIN_NTRAIN="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$DATASET_NAME" || -z "$SPLITS_DIR" || -z "$PREDICTIONS_DIR" || -z "$MODEL_DIR" || -z "$SEED" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ "$ENABLED" != "1" || "$DATASET_NAME" != "RR" ]]; then
    exit 0
fi

mkdir -p "$PREDICTIONS_DIR"

shopt -s nullglob
train_csvs=("$SPLITS_DIR"/train_*.csv)
shopt -u nullglob

for train_csv in "${train_csvs[@]}"; do
    train_file="$(basename "$train_csv")"
    num_train_samples="${train_file#train_}"
    num_train_samples="${num_train_samples%.csv}"

    if (( num_train_samples < MIN_NTRAIN )); then
        continue
    fi

    plm_model_dir="$MODEL_DIR/pLM_encoder_${num_train_samples}_supervised"
    prediction_file="$PREDICTIONS_DIR/test.embedding.plm_supervised.ntrain_${num_train_samples}.predictions.h5"

    echo "Running RR supervised PLM finetuning for ntrain=${num_train_samples}"
    python src/plm/train_supervised.py \
        --train_csv "$train_csv" \
        --test_csv "$SPLITS_DIR/test.csv" \
        --folder_params "$plm_model_dir" \
        --epochs 50 \
        --seed "$SEED" \
        --bf16

    echo "Running RR supervised PLM predictions for ntrain=${num_train_samples}"
    python src/predict_from_plm_supervised_freeze.py \
        --model_dir "$plm_model_dir" \
        --train_csv "$train_csv" \
        --query "$SPLITS_DIR/test.csv" \
        --output "$prediction_file" \
        --info "train-${num_train_samples}-plm_supervised" \
        --bf16
done
