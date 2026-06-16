#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/08_predict_generated_samples.sh \
    --source_csv <path> \
    --splits_dir <path> \
    --embeddings_dir <path> \
    --model_dir <path> \
    --rbm_model_path <path>

Embeds generated RBM samples and predicts labels for them.
EOF
}

SOURCE_CSV=""
SPLITS_DIR=""
EMBEDDINGS_DIR=""
MODEL_DIR=""
RBM_MODEL_PATH=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source_csv) SOURCE_CSV="$2"; shift 2 ;;
        --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
        --embeddings_dir) EMBEDDINGS_DIR="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --rbm_model_path|--rbm_checkpoint) RBM_MODEL_PATH="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$SOURCE_CSV" || -z "$SPLITS_DIR" || -z "$EMBEDDINGS_DIR" || -z "$MODEL_DIR" || -z "$RBM_MODEL_PATH" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ ! -f "$SOURCE_CSV" ]]; then
    echo "source_csv does not exist: $SOURCE_CSV" >&2
    exit 1
fi

if [[ ! -d "$SPLITS_DIR" ]]; then
    echo "splits_dir does not exist: $SPLITS_DIR" >&2
    exit 1
fi

if [[ ! -d "$EMBEDDINGS_DIR" ]]; then
    echo "embeddings_dir does not exist: $EMBEDDINGS_DIR" >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "model_dir does not exist: $MODEL_DIR" >&2
    exit 1
fi

if [[ ! -f "$RBM_MODEL_PATH" ]]; then
    echo "rbm_model_path does not exist: $RBM_MODEL_PATH" >&2
    exit 1
fi

shopt -s nullglob
label_dirs=("$MODEL_DIR"/RBM_labels_*)
shopt -u nullglob

if [[ ${#label_dirs[@]} -eq 0 ]]; then
    echo "No RBM_labels_* folders found in: $MODEL_DIR" >&2
    exit 1
fi

num_runs=0

for label_dir in "${label_dirs[@]}"; do
    [[ -d "$label_dir" ]] || continue

    label_name="$(basename "$label_dir")"
    samples_csv="$label_dir/samples.csv"

    if [[ ! -f "$samples_csv" ]]; then
        echo "Skipping $label_name: missing samples.csv" >&2
        continue
    fi

    if [[ "$label_name" == "RBM_labels_all_true" ]]; then
        embedding_h5="$label_dir/samples.embedding.true.h5"

        python src/encoding/onehot_encoding.py \
            --query "$samples_csv" \
            --output "$embedding_h5" \
            --info "generated-0-true" \
            --column_sequences sequence_align \
            --column_labels true_label \
            --column_headers header

        python src/prediction/predict_from_csv_true.py \
            --train_csv "$SOURCE_CSV" \
            --query_dir "$label_dir" \
            --column_sequences sequence_align \
            --column_labels label \
            --column_headers header \
            --info "source_csv=$SOURCE_CSV"

        num_runs=$((num_runs + 1))
        continue
    fi

    if [[ ! "$label_name" =~ ^RBM_labels_([0-9]+)_(.+)$ ]]; then
        echo "Skipping unexpected folder name: $label_name" >&2
        continue
    fi

    num_train_samples="${BASH_REMATCH[1]}"
    model="${BASH_REMATCH[2]}"
    label_column="${model}_label"

    case "$model" in
        onehot)
            embedding_h5="$label_dir/samples.embedding.onehot.h5"
            train_h5="$EMBEDDINGS_DIR/train_${num_train_samples}.embedding.onehot.h5"

            if [[ ! -f "$train_h5" ]]; then
                echo "Skipping $label_name: missing train embedding $train_h5" >&2
                continue
            fi

            python src/encoding/onehot_encoding.py \
                --query "$samples_csv" \
                --output "$embedding_h5" \
                --info "generated-${num_train_samples}-onehot" \
                --column_sequences sequence_align \
                --column_labels "$label_column" \
                --column_headers header

            python src/predict_from_embeddings.py \
                --train_h5 "$train_h5" \
                --test_h5 "$embedding_h5" \
                --output_path "$label_dir/samples.predictions.onehot.h5"
            ;;
        rbm)
            embedding_h5="$label_dir/samples.embedding.rbm.h5"
            train_h5="$EMBEDDINGS_DIR/train_${num_train_samples}.embedding.rbm.h5"

            if [[ ! -f "$train_h5" ]]; then
                echo "Skipping $label_name: missing train embedding $train_h5" >&2
                continue
            fi

            python src/encoding/rbm_encoding.py \
                --model "$RBM_MODEL_PATH" \
                --query "$samples_csv" \
                --output "$embedding_h5" \
                --info "generated-${num_train_samples}-rbm" \
                --column_sequences sequence_align \
                --column_labels "$label_column" \
                --column_headers header

            python src/predict_from_embeddings.py \
                --train_h5 "$train_h5" \
                --test_h5 "$embedding_h5" \
                --output_path "$label_dir/samples.predictions.rbm.h5"
            ;;
        plm)
            embedding_h5="$label_dir/samples.embedding.plm.h5"
            train_h5="$EMBEDDINGS_DIR/train_${num_train_samples}.embedding.plm.h5"

            if [[ ! -f "$train_h5" ]]; then
                echo "Skipping $label_name: missing train embedding $train_h5" >&2
                continue
            fi

            python src/encoding/plm_encoding.py \
                --query "$samples_csv" \
                --output "$embedding_h5" \
                --info "generated-${num_train_samples}-plm" \
                --column_sequences sequence_align \
                --column_labels "$label_column" \
                --column_headers header \
                --bf16

            python src/predict_from_embeddings.py \
                --train_h5 "$train_h5" \
                --test_h5 "$embedding_h5" \
                --output_path "$label_dir/samples.predictions.plm.h5"
            ;;
        plm_supervised)
            train_csv="$SPLITS_DIR/train_${num_train_samples}.csv"
            plm_model_dir="$MODEL_DIR/pLM_encoder_${num_train_samples}_supervised"

            if [[ ! -f "$train_csv" ]]; then
                echo "Skipping $label_name: missing train CSV $train_csv" >&2
                continue
            fi

            if [[ ! -d "$plm_model_dir" ]]; then
                echo "Skipping $label_name: missing supervised PLM model $plm_model_dir" >&2
                continue
            fi

            python src/predict_from_plm_supervised_freeze.py \
                --model_dir "$plm_model_dir" \
                --train_csv "$train_csv" \
                --query "$samples_csv" \
                --output "$label_dir/samples.predictions.plm_supervised.h5" \
                --info "generated-${num_train_samples}-plm_supervised" \
                --column_sequences sequence_align \
                --column_labels "$label_column" \
                --column_headers header \
                --bf16
            ;;
        *)
            echo "Skipping unsupported generated-sample model: $label_name" >&2
            continue
            ;;
    esac

    num_runs=$((num_runs + 1))
done

if [[ $num_runs -eq 0 ]]; then
    echo "No generated-sample prediction runs were executed." >&2
    exit 1
fi

echo "Completed $num_runs generated-sample prediction runs."
