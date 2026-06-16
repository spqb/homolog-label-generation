#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/03_predict_test_embeddings.sh \
    --embeddings_dir <path> \
    --predictions_dir <path>

Pairs train/test embedding HDF5 files and writes test prediction HDF5 files.
EOF
}

EMBEDDINGS_DIR=""
PREDICTIONS_DIR=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --embeddings_dir) EMBEDDINGS_DIR="$2"; shift 2 ;;
        --predictions_dir) PREDICTIONS_DIR="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$EMBEDDINGS_DIR" || -z "$PREDICTIONS_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ ! -d "$EMBEDDINGS_DIR" ]]; then
    echo "embeddings_dir does not exist: $EMBEDDINGS_DIR" >&2
    exit 1
fi

mkdir -p "$PREDICTIONS_DIR"

shopt -s nullglob
h5_files=("$EMBEDDINGS_DIR"/*.h5)
shopt -u nullglob

if [[ ${#h5_files[@]} -eq 0 ]]; then
    echo "No .h5 files found in: $EMBEDDINGS_DIR" >&2
    exit 1
fi

declare -A TEST_BY_MODEL
declare -A TEST_BY_MODEL_AND_N

for file_path in "${h5_files[@]}"; do
    file_name="$(basename "$file_path")"
    if [[ "$file_name" =~ ^test\.embedding\.supervised\.([0-9]+)\.h5$ ]]; then
        ntrain="${BASH_REMATCH[1]}"
        key="supervised:${ntrain}"
        if [[ -n "${TEST_BY_MODEL_AND_N[$key]:-}" ]]; then
            echo "Warning: multiple supervised test files for ntrain '$ntrain'. Using: ${TEST_BY_MODEL_AND_N[$key]}" >&2
            echo "         skipping: $file_path" >&2
        else
            TEST_BY_MODEL_AND_N[$key]="$file_path"
        fi
        continue
    fi

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

if [[ ${#TEST_BY_MODEL[@]} -eq 0 && ${#TEST_BY_MODEL_AND_N[@]} -eq 0 ]]; then
    echo "No test embedding files found in: $EMBEDDINGS_DIR" >&2
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

    if [[ "$model" == "supervised" ]]; then
        test_path="${TEST_BY_MODEL_AND_N["supervised:${ntrain}"]:-}"
    else
        test_path="${TEST_BY_MODEL[$model]:-}"
    fi

    if [[ -z "$test_path" ]]; then
        echo "Skipping $train_name: no compatible test file for model '$model'." >&2
        continue
    fi

    output_file="$PREDICTIONS_DIR/test.embedding.${model}.ntrain_${ntrain}.predictions.h5"

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

echo "Completed $num_runs test prediction runs."
