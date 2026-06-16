#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/05_prepare_rbm_datasets.sh \
    --predictions_dir <path> \
    --source_csv <path> \
    --output_dir <path>
EOF
}

PREDICTIONS_DIR=""
SOURCE_CSV=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --predictions_dir) PREDICTIONS_DIR="$2"; shift 2 ;;
        --source_csv) SOURCE_CSV="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$PREDICTIONS_DIR" || -z "$SOURCE_CSV" || -z "$OUTPUT_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

python src/prepare_rbm_dataset.py \
    --input_dir "$PREDICTIONS_DIR" \
    --csv_pool "$SOURCE_CSV" \
    --output_dir "$OUTPUT_DIR"
