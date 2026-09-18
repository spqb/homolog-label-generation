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
    [--samples_dir <path>] \
    --rbm_model_path <path> \
    [--methods <method-id> ...] \
    [--train_sizes <N> ...] \
    [--skip_existing] \
    [--dry_run] \
    [--esm2_model <model-id>] \
    [--max_length <residues; default 512>] \
    [--ankh_model <model-id>] \
    [--esmc_model <model-id>]

Embeds generated RBM samples and predicts labels for them.
This stage operates only on the balanced self-consistency samples created by stage 7.
EOF
}

SOURCE_CSV=""
SPLITS_DIR=""
EMBEDDINGS_DIR=""
MODEL_DIR=""
SAMPLES_DIR=""
RBM_MODEL_PATH=""
ESM2_MODEL="facebook/esm2_t33_650M_UR50D"
ESMC_MODEL="biohub/ESMC-600M"
ANKH_MODEL="ElnaggarLab/ankh-base"
MAX_LENGTH=512
METHODS=()
TRAIN_SIZES=()
SKIP_EXISTING=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --source_csv) SOURCE_CSV="$2"; shift 2 ;;
        --splits_dir) SPLITS_DIR="$2"; shift 2 ;;
        --embeddings_dir) EMBEDDINGS_DIR="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --samples_dir) SAMPLES_DIR="$2"; shift 2 ;;
        --rbm_model_path|--rbm_checkpoint) RBM_MODEL_PATH="$2"; shift 2 ;;
        --methods)
            shift; METHODS=()
            while [[ $# -gt 0 && "$1" != --* ]]; do METHODS+=("$1"); shift; done
            [[ ${#METHODS[@]} -gt 0 ]] || { echo "--methods requires method ids" >&2; exit 1; }
            ;;
        --train_sizes)
            shift; TRAIN_SIZES=()
            while [[ $# -gt 0 && "$1" != --* ]]; do TRAIN_SIZES+=("$1"); shift; done
            [[ ${#TRAIN_SIZES[@]} -gt 0 ]] || { echo "--train_sizes requires integers" >&2; exit 1; }
            ;;
        --skip_existing) SKIP_EXISTING=1; shift ;;
        --dry_run) DRY_RUN=1; shift ;;
        --esm2_model) ESM2_MODEL="$2"; shift 2 ;;
        --esmc_model) ESMC_MODEL="$2"; shift 2 ;;
        --max_length) MAX_LENGTH="$2"; shift 2 ;;
        --ankh_model) ANKH_MODEL="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

[[ "$MAX_LENGTH" =~ ^[1-9][0-9]*$ ]] || { echo "--max_length must be a positive integer." >&2; exit 1; }

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

DATASETS_DIR="$(dirname "$EMBEDDINGS_DIR")/datasets_for_rbm"
shopt -s nullglob
dataset_files=("$DATASETS_DIR"/rbm_dataset_ntrain_*.csv)
shopt -u nullglob
[[ ${#dataset_files[@]} -gt 0 ]] || { echo "No RBM datasets found in $DATASETS_DIR" >&2; exit 1; }
largest_dataset="$(printf '%s\n' "${dataset_files[@]}" | sort -V | tail -n 1)"

num_runs=0
num_skipped=0

contains() {
    local wanted="$1"; shift
    local value
    for value in "$@"; do [[ "$value" == "$wanted" ]] && return 0; done
    return 1
}

run_command() {
    printf 'Running:'
    printf ' %q' "$@"
    printf '\n'
    [[ $DRY_RUN -eq 1 ]] || "$@"
}

for label_dir in "${label_dirs[@]}"; do
    [[ -d "$label_dir" ]] || continue

    label_name="$(basename "$label_dir")"
    artifact_dir="${SAMPLES_DIR:-$label_dir}"
    [[ -z "$SAMPLES_DIR" ]] || artifact_dir="$SAMPLES_DIR/$label_name"
    [[ $DRY_RUN -eq 1 ]] || mkdir -p "$artifact_dir"
    samples_csv="$artifact_dir/samples.self_consistency.csv"

    if [[ "$label_name" == "RBM_labels_all_true_full" ]]; then
        [[ ${#METHODS[@]} -eq 0 ]] || contains true "${METHODS[@]}" || continue
        prediction_h5="$artifact_dir/samples.self_consistency.predictions.true.h5"
        if [[ ! -f "$samples_csv" ]]; then
            echo "Skipping $label_name: missing self-consistency samples" >&2
            continue
        fi
        if [[ $SKIP_EXISTING -eq 1 && -s "$prediction_h5" && "$prediction_h5" -nt "$samples_csv" ]]; then
            echo "Skipping completed predictions: $prediction_h5"
            num_skipped=$((num_skipped + 1))
            continue
        fi
        embedding_h5="$artifact_dir/samples.self_consistency.embedding.true.h5"

        if [[ $SKIP_EXISTING -eq 0 || ! -s "$embedding_h5" || "$samples_csv" -nt "$embedding_h5" ]]; then
            run_command python src/encoding/onehot_encoding.py \
                --query "$samples_csv" \
                --output "$embedding_h5" \
                --info "balanced-self-consistency-true" \
                --column_sequences sequence_align \
                --column_labels target_label \
                --column_headers header
        fi

        run_command python src/prediction/predict_from_csv_true.py \
            --train_csv "$largest_dataset" \
            --query_h5 "$embedding_h5" \
            --output_path "$prediction_h5" \
            --column_sequences sequence_align \
            --column_labels true_label \
            --column_headers header \
            --info "train_union=$largest_dataset"

        num_runs=$((num_runs + 1))
        continue
    fi

    if [[ ! "$label_name" =~ ^RBM_labels_([0-9]+)_(.+)$ ]]; then
        [[ "$label_name" == "RBM_labels_all_true" ]] && continue
        echo "Skipping unexpected folder name: $label_name" >&2
        continue
    fi

    num_train_samples="${BASH_REMATCH[1]}"
    model="${BASH_REMATCH[2]}"
    label_column="target_label"

    if [[ ${#METHODS[@]} -gt 0 ]] && ! contains "$model" "${METHODS[@]}"; then
        continue
    fi
    if [[ ${#TRAIN_SIZES[@]} -gt 0 ]] && ! contains "$num_train_samples" "${TRAIN_SIZES[@]}"; then
        continue
    fi
    if [[ ! -f "$samples_csv" ]]; then
        echo "Skipping $label_name: missing self-consistency samples" >&2
        continue
    fi

    prediction_h5="$artifact_dir/samples.self_consistency.predictions.${model}.h5"
    if [[ $SKIP_EXISTING -eq 1 && -s "$prediction_h5" && "$prediction_h5" -nt "$samples_csv" ]]; then
        echo "Skipping completed predictions: $prediction_h5"
        num_skipped=$((num_skipped + 1))
        continue
    fi

    case "$model" in
        onehot)
            embedding_h5="$artifact_dir/samples.self_consistency.embedding.onehot.h5"
            train_h5="$EMBEDDINGS_DIR/train_${num_train_samples}.embedding.onehot.h5"

            if [[ ! -f "$train_h5" ]]; then
                echo "Skipping $label_name: missing train embedding $train_h5" >&2
                continue
            fi

            if [[ $SKIP_EXISTING -eq 0 || ! -s "$embedding_h5" || "$samples_csv" -nt "$embedding_h5" ]]; then
                run_command python src/encoding/onehot_encoding.py \
                --query "$samples_csv" \
                --output "$embedding_h5" \
                --info "balanced-self-consistency-${num_train_samples}-onehot" \
                --column_sequences sequence_align \
                --column_labels "$label_column" \
                --column_headers header
            fi

            run_command python src/predict_from_embeddings.py \
                --train_h5 "$train_h5" \
                --test_h5 "$embedding_h5" \
                --output_path "$prediction_h5"
            ;;
        rbm)
            embedding_h5="$artifact_dir/samples.self_consistency.embedding.rbm.h5"
            train_h5="$EMBEDDINGS_DIR/train_${num_train_samples}.embedding.rbm.h5"

            if [[ ! -f "$train_h5" ]]; then
                echo "Skipping $label_name: missing train embedding $train_h5" >&2
                continue
            fi

            if [[ $SKIP_EXISTING -eq 0 || ! -s "$embedding_h5" || "$samples_csv" -nt "$embedding_h5" ]]; then
                run_command python src/encoding/rbm_encoding.py \
                --model "$RBM_MODEL_PATH" \
                --query "$samples_csv" \
                --output "$embedding_h5" \
                --info "balanced-self-consistency-${num_train_samples}-rbm" \
                --column_sequences sequence_align \
                --column_labels "$label_column" \
                --column_headers header
            fi

            run_command python src/predict_from_embeddings.py \
                --train_h5 "$train_h5" \
                --test_h5 "$embedding_h5" \
                --output_path "$prediction_h5"
            ;;
        esm2|esmc|ankh)
            embedding_h5="$artifact_dir/samples.self_consistency.embedding.${model}.h5"
            train_h5="$EMBEDDINGS_DIR/train_${num_train_samples}.embedding.${model}.h5"

            if [[ ! -f "$train_h5" ]]; then
                echo "Skipping $label_name: missing train embedding $train_h5" >&2
                continue
            fi

            if [[ "$model" == "esm2" ]]; then
                model_id="$ESM2_MODEL"
            elif [[ "$model" == "ankh" ]]; then
                model_id="$ANKH_MODEL"
            else
                model_id="$ESMC_MODEL"
            fi

            if [[ $SKIP_EXISTING -eq 0 || ! -s "$embedding_h5" || "$samples_csv" -nt "$embedding_h5" ]]; then
                run_command python src/encoding/plm_encoding.py \
                --query "$samples_csv" \
                --output "$embedding_h5" \
                --info "balanced-self-consistency-${num_train_samples}-${model}" \
                --model "$model_id" \
                --max_length "$MAX_LENGTH" \
                --column_sequences sequence_align \
                --column_labels "$label_column" \
                --column_headers header \
                --bf16
            fi

            run_command python src/predict_from_embeddings.py \
                --train_h5 "$train_h5" \
                --test_h5 "$embedding_h5" \
                --output_path "$prediction_h5"
            ;;
        esm2_supervised)
            train_csv="$SPLITS_DIR/train_${num_train_samples}.csv"
            esm2_model_dir="$MODEL_DIR/esm2_encoder_${num_train_samples}_supervised"

            if [[ ! -f "$train_csv" ]]; then
                echo "Skipping $label_name: missing train CSV $train_csv" >&2
                continue
            fi

            if [[ ! -d "$esm2_model_dir" ]]; then
                echo "Skipping $label_name: missing supervised ESM2 model $esm2_model_dir" >&2
                continue
            fi

            run_command python src/predict_from_plm_supervised_freeze.py \
                --model_dir "$esm2_model_dir" \
                --max_length "$MAX_LENGTH" \
                --train_csv "$train_csv" \
                --query "$samples_csv" \
                --output "$prediction_h5" \
                --info "balanced-self-consistency-${num_train_samples}-esm2_supervised" \
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

if [[ $num_runs -eq 0 && $num_skipped -eq 0 ]]; then
    echo "No generated-sample prediction runs were executed." >&2
    exit 1
fi

echo "Completed $num_runs generated-sample prediction runs; completed artifacts skipped: $num_skipped."
