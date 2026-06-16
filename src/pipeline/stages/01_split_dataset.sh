#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/01_split_dataset.sh \
    --source_csv <path> \
    --seed <int> \
    --t1 <float> \
    --output_dir <path>
EOF
}

SOURCE_CSV=""
SEED=""
T1=""
OUTPUT_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source_csv) SOURCE_CSV="$2"; shift 2 ;;
        --seed) SEED="$2"; shift 2 ;;
        --t1) T1="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$SOURCE_CSV" || -z "$SEED" || -z "$T1" || -z "$OUTPUT_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

python src/split_train_test.py \
    --source_csv "$SOURCE_CSV" \
    --seed "$SEED" \
    --t1 "$T1" \
    --output_dir "$OUTPUT_DIR"
