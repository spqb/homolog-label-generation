#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
    src/scripts/create_embeddings.sh --input_folder <path> --embedding_model <onehot|rbm|plm> --output_folder <path> [--checkpoint <path>] [--column_sequences <name>] [--column_labels <name>] [--column_headers <name>] [--batch_size <int>] [--max_length <int>] [--bf16]

Arguments:
  --input_folder     Folder containing query CSV files (e.g. train_100.csv, test.csv).
  --embedding_model  Embedding model to run: onehot, rbm, or plm.
  --output_folder    Folder where embeddings will be written.
  --checkpoint       Required for rbm and plm. Path to model/checkpoint to load.
    --column_sequences Optional sequence-column name override for input CSV files.
    --column_labels    Optional label-column name override for input CSV files.
    --column_headers   Optional header-column name override for input CSV files.
    --batch_size       Optional (PLM only): batch size for embedding.
    --max_length       Optional (PLM only): max sequence length.
    --bf16             Optional (PLM only): enable bfloat16 autocast.
  -h, --help         Show this help message.

Output naming:
  <output_folder>/<input_basename>.embedding.<model>.h5

Info format:
  train-<N>-<model> for train_<N>.csv
  test-0-<model> for test.csv
EOF
}

INPUT_FOLDER=""
EMBEDDING_MODEL=""
CHECKPOINT=""
OUTPUT_FOLDER=""
COLUMN_SEQUENCES=""
COLUMN_LABELS=""
COLUMN_HEADERS=""
PLM_BATCH_SIZE=""
PLM_MAX_LENGTH=""
PLM_BF16=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --input_folder)
            INPUT_FOLDER="$2"
            shift 2
            ;;
        --embedding_model)
            EMBEDDING_MODEL="$2"
            shift 2
            ;;
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --output_folder)
            OUTPUT_FOLDER="$2"
            shift 2
            ;;
        --column_sequences)
            COLUMN_SEQUENCES="$2"
            shift 2
            ;;
        --column_labels)
            COLUMN_LABELS="$2"
            shift 2
            ;;
        --column_headers)
            COLUMN_HEADERS="$2"
            shift 2
            ;;
        --batch_size)
            PLM_BATCH_SIZE="$2"
            shift 2
            ;;
        --max_length)
            PLM_MAX_LENGTH="$2"
            shift 2
            ;;
        --bf16)
            PLM_BF16=true
            shift
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

if [[ -z "$INPUT_FOLDER" || -z "$EMBEDDING_MODEL" || -z "$OUTPUT_FOLDER" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ ! -d "$INPUT_FOLDER" ]]; then
    echo "Input folder does not exist: $INPUT_FOLDER" >&2
    exit 1
fi

if [[ "$EMBEDDING_MODEL" != "plm" ]]; then
    if [[ -n "$PLM_BATCH_SIZE" || -n "$PLM_MAX_LENGTH" || "$PLM_BF16" == true ]]; then
        echo "--batch_size, --max_length, and --bf16 can only be used with --embedding_model plm." >&2
        exit 1
    fi
fi

case "$EMBEDDING_MODEL" in
    onehot)
        SCRIPT_PATH="src/encoding/onehot_encoding.py"
        ;;
    rbm)
        SCRIPT_PATH="src/encoding/rbm_encoding.py"
        if [[ -z "$CHECKPOINT" ]]; then
            echo "--checkpoint is required when --embedding_model rbm." >&2
            exit 1
        fi
        ;;
    plm)
        SCRIPT_PATH="src/encoding/plm_encoding.py"
        if [[ -z "$CHECKPOINT" ]]; then
            echo "--checkpoint is required when --embedding_model plm." >&2
            exit 1
        fi
        ;;
    *)
        echo "Invalid --embedding_model '$EMBEDDING_MODEL'. Use onehot, rbm, or plm." >&2
        exit 1
        ;;
esac

mkdir -p "$OUTPUT_FOLDER"

shopt -s nullglob
csv_files=("$INPUT_FOLDER"/*.csv)
shopt -u nullglob

if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No CSV files found in: $INPUT_FOLDER" >&2
    exit 1
fi

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

    info="${split_name}-${num_train}-${EMBEDDING_MODEL}"
    output_file="$OUTPUT_FOLDER/${file_name%.csv}.embedding.${EMBEDDING_MODEL}.h5"

    echo "Embedding $query_file"
    echo "  -> $output_file"
    echo "  info=$info"

    extra_args=()
    if [[ -n "$COLUMN_SEQUENCES" ]]; then
        extra_args+=(--column_sequences "$COLUMN_SEQUENCES")
    fi
    if [[ -n "$COLUMN_LABELS" ]]; then
        extra_args+=(--column_labels "$COLUMN_LABELS")
    fi
    if [[ -n "$COLUMN_HEADERS" ]]; then
        extra_args+=(--column_headers "$COLUMN_HEADERS")
    fi

    plm_extra_args=()
    if [[ -n "$PLM_BATCH_SIZE" ]]; then
        plm_extra_args+=(--batch_size "$PLM_BATCH_SIZE")
    fi
    if [[ -n "$PLM_MAX_LENGTH" ]]; then
        plm_extra_args+=(--max_length "$PLM_MAX_LENGTH")
    fi
    if [[ "$PLM_BF16" == true ]]; then
        plm_extra_args+=(--bf16)
    fi

    if [[ "$EMBEDDING_MODEL" == "onehot" ]]; then
        python "$SCRIPT_PATH" \
            --query "$query_file" \
            --output "$output_file" \
            --info "$info" \
            "${extra_args[@]}"
    elif [[ "$EMBEDDING_MODEL" == "rbm" ]]; then
        python "$SCRIPT_PATH" \
            --query "$query_file" \
            --output "$output_file" \
            --info "$info" \
            --model "$CHECKPOINT" \
            "${extra_args[@]}"
    else
        python "$SCRIPT_PATH" \
            --query "$query_file" \
            --output "$output_file" \
            --info "$info" \
            --checkpoint "$CHECKPOINT" \
            "${extra_args[@]}" \
            "${plm_extra_args[@]}"
    fi
done

echo "All embeddings created in: $OUTPUT_FOLDER"
