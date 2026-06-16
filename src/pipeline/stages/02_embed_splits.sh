#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/02_embed_splits.sh \
    --splits_dir <path> \
    --output_dir <path> \
    --rbm_model_path <path>

Creates onehot, RBM, and PLM embeddings for train_<N>.csv and test.csv files.
EOF
}

SPLITS_DIR=""
OUTPUT_DIR=""
RBM_MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
        --output_dir) OUTPUT_DIR="$2"; shift 2 ;;
        --rbm_model_path|--rbm_embedding_model) RBM_MODEL_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$SPLITS_DIR" || -z "$OUTPUT_DIR" || -z "$RBM_MODEL_PATH" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ ! -d "$SPLITS_DIR" ]]; then
    echo "splits_dir does not exist: $SPLITS_DIR" >&2
    exit 1
fi

if [[ ! -f "$RBM_MODEL_PATH" ]]; then
    echo "rbm_model_path does not exist: $RBM_MODEL_PATH" >&2
    exit 1
fi

mkdir -p "$OUTPUT_DIR"

shopt -s nullglob
csv_files=("$SPLITS_DIR"/*.csv)
shopt -u nullglob

if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No CSV files found in: $SPLITS_DIR" >&2
    exit 1
fi

run_embedding_model() {
    local embedding_model="$1"
    local query_file file_name split_name num_train info output_file sequence_column script_path

    case "$embedding_model" in
        onehot)
            script_path="src/encoding/onehot_encoding.py"
            sequence_column="sequence_align"
            ;;
        rbm)
            script_path="src/encoding/rbm_encoding.py"
            sequence_column="sequence_align"
            ;;
        plm)
            script_path="src/encoding/plm_encoding.py"
            sequence_column="sequence"
            ;;
        *)
            echo "Invalid embedding model: $embedding_model" >&2
            exit 1
            ;;
    esac

    for query_file in "${csv_files[@]}"; do
        file_name="$(basename "$query_file")"

        if [[ "$file_name" =~ ^train_([0-9]+)\.csv$ ]]; then
            split_name="train"
            num_train="${BASH_REMATCH[1]}"
        elif [[ "$file_name" == "test.csv" ]]; then
            split_name="test"
            num_train="0"
        else
            echo "Skipping unsupported file name: $file_name" >&2
            continue
        fi

        info="${split_name}-${num_train}-${embedding_model}"
        output_file="$OUTPUT_DIR/${file_name%.csv}.embedding.${embedding_model}.h5"

        echo "Embedding $query_file"
        echo "  model=$embedding_model"
        echo "  out=$output_file"

        if [[ "$embedding_model" == "onehot" ]]; then
            python "$script_path" \
                --query "$query_file" \
                --output "$output_file" \
                --info "$info" \
                --column_sequences "$sequence_column"
        elif [[ "$embedding_model" == "rbm" ]]; then
            python "$script_path" \
                --query "$query_file" \
                --output "$output_file" \
                --info "$info" \
                --model "$RBM_MODEL_PATH" \
                --column_sequences "$sequence_column"
        else
            python "$script_path" \
                --query "$query_file" \
                --output "$output_file" \
                --info "$info" \
                --column_sequences "$sequence_column" \
                --bf16
        fi
    done
}

run_embedding_model onehot
run_embedding_model rbm
run_embedding_model plm

echo "All split embeddings created in: $OUTPUT_DIR"
