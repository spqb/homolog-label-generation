#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/scripts/run_predictions_from_embeddings.sh --input_folder <path> --output_folder <path>

Arguments:
  --input_folder   Folder containing embedding .h5 files generated from train/test splits.
  --output_folder  Folder where prediction .h5 files will be saved.
  -h, --help       Show this help message.

Expected embedding names:
  train_<N>.embedding.<model>.h5
  test.embedding.<model>.h5

Compatibility rule:
  A train file is compatible with a test file if they share the same <model>.
EOF
}

INPUT_FOLDER=""
OUTPUT_FOLDER=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER="$2"
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

if [[ -z "$INPUT_FOLDER" || -z "$OUTPUT_FOLDER" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ ! -d "$INPUT_FOLDER" ]]; then
    echo "Input folder does not exist: $INPUT_FOLDER" >&2
    exit 1
fi

mkdir -p "$OUTPUT_FOLDER"

shopt -s nullglob
h5_files=("$INPUT_FOLDER"/*.h5)
shopt -u nullglob

if [[ ${#h5_files[@]} -eq 0 ]]; then
    echo "No .h5 files found in: $INPUT_FOLDER" >&2
    exit 1
fi

declare -A TEST_BY_MODEL

for file_path in "${h5_files[@]}"; do
    file_name="$(basename "$file_path")"
    if [[ "$file_name" =~ ^test\.embedding\.([^.]+)\.h5$ ]]; then
        model="${BASH_REMATCH[1]}"
        if [[ -n "${TEST_BY_MODEL[$model]:-}" ]]; then
            echo "Warning: multiple test files for model '$model'. Using: ${TEST_BY_MODEL[$model]}" >&2
            echo "         skipping: $file_path" >&2
        else
            TEST_BY_MODEL[$model]="$file_path"
        fi
    fi
done

if [[ ${#TEST_BY_MODEL[@]} -eq 0 ]]; then
    echo "No test embedding files found matching 'test.embedding.<model>.h5' in: $INPUT_FOLDER" >&2
    exit 1
fi

num_runs=0
for train_path in "${h5_files[@]}"; do
    train_name="$(basename "$train_path")"

    if [[ ! "$train_name" =~ ^train_([0-9]+)\.embedding\.([^.]+)\.h5$ ]]; then
        continue
    fi

    ntrain="${BASH_REMATCH[1]}"
    model="${BASH_REMATCH[2]}"

    test_path="${TEST_BY_MODEL[$model]:-}"
    if [[ -z "$test_path" ]]; then
        echo "Skipping $train_name: no compatible test file for model '$model'." >&2
        continue
    fi

    output_file="$OUTPUT_FOLDER/test.embedding.${model}.ntrain_${ntrain}.predictions.h5"

    echo "Running predictions for model=$model, ntrain=$ntrain"
    echo "  train: $train_path"
    echo "  test : $test_path"
    echo "  out  : $output_file"

    python src/predict_from_embeddings.py \
        --train_h5 "$train_path" \
        --test_h5 "$test_path" \
        --output_path "$output_file"

    num_runs=$((num_runs + 1))
done

if [[ $num_runs -eq 0 ]]; then
    echo "No compatible train/test embedding pairs were found." >&2
    exit 1
fi

echo "Completed $num_runs prediction runs. Results saved in: $OUTPUT_FOLDER"
