#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/05_prepare_rbm_datasets.sh \
    --predictions_dir <path> \
    --source_csv <path> \
    --output_dir <path> \
    [--models <model-id> ...] \
    [--train_sizes <N> ...] \
    [--merge_existing]
EOF
}

PREDICTIONS_DIR=""
SOURCE_CSV=""
OUTPUT_DIR=""
MODELS=()
TRAIN_SIZES=()
MERGE_EXISTING=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictions_dir) PREDICTIONS_DIR="$2"; shift 2 ;;
        --source_csv) SOURCE_CSV="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --models)
            shift; MODELS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do MODELS+=("$1"); shift; done
            [[ ${#MODELS[@]} -gt 0 ]] || { echo "--models requires model ids" >&2; exit 1; }
            ;;
        --train_sizes)
            shift; TRAIN_SIZES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do TRAIN_SIZES+=("$1"); shift; done
            [[ ${#TRAIN_SIZES[@]} -gt 0 ]] || { echo "--train_sizes requires integers" >&2; exit 1; }
            ;;
        --merge_existing) MERGE_EXISTING=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$PREDICTIONS_DIR" || -z "$SOURCE_CSV" || -z "$OUTPUT_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

cmd=(python src/prepare_rbm_dataset.py \
    --input_dir "$PREDICTIONS_DIR" \
    --csv_pool "$SOURCE_CSV" \
    --output_dir "$OUTPUT_DIR")
[[ ${#MODELS[@]} -eq 0 ]] || cmd+=(--models "${MODELS[@]}")
[[ ${#TRAIN_SIZES[@]} -eq 0 ]] || cmd+=(--train_sizes "${TRAIN_SIZES[@]}")
[[ $MERGE_EXISTING -eq 0 ]] || cmd+=(--merge_existing)
"${cmd[@]}"
