#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/06_train_conditioned_rbms.sh \
    --datasets_dir <path> \
    --model_dir <path> \
    [--annadca_bin annadca] \
    [--dry_run]

Trains one conditioned RBM for each supported label column in each
rbm_dataset_ntrain_*.csv.
EOF
}

DATASETS_DIR=""
MODEL_DIR=""
ANNADCA_BIN="annadca"
DRY_RUN=0
SEQUENCE_COLUMN="sequence_align"
NAME_COLUMN="header"
NEPOCHS="50000"
EXTRA_ARGS=(--hidden 500 --no_reweighting --lr 0.005)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets_dir) DATASETS_DIR="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --annadca_bin) ANNADCA_BIN="$2"; shift 2 ;;
        --dry_run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

if [[ -z "$DATASETS_DIR" || -z "$MODEL_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ ! -d "$DATASETS_DIR" ]]; then
    echo "datasets_dir does not exist: $DATASETS_DIR" >&2
    exit 1
fi

if [[ $DRY_RUN -eq 0 ]] && ! command -v "$ANNADCA_BIN" >/dev/null 2>&1; then
    echo "annadca executable not found: $ANNADCA_BIN" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

mapfile -t csv_files < <(find "$DATASETS_DIR" -type f -name 'rbm_dataset_ntrain_*.csv' | sort -V)

if [[ ${#csv_files[@]} -eq 0 ]]; then
    echo "No rbm_dataset_ntrain_*.csv files found under $DATASETS_DIR" >&2
    exit 1
fi

trained_true_label=0
num_runs=0

for csv_path in "${csv_files[@]}"; do
    filename="$(basename "$csv_path")"
    if [[ "$filename" =~ rbm_dataset_ntrain_([0-9]+)\.csv$ ]]; then
        ntrain="${BASH_REMATCH[1]}"
    else
        echo "Skipping unexpected filename format: $csv_path" >&2
        continue
    fi

    header_line="$(head -n 1 "$csv_path")"
    IFS=',' read -r -a columns <<< "$header_line"

    label_cols=()
    for column in "${columns[@]}"; do
        if [[ "$column" == *_label || "$column" == "train_only" ]]; then
            label_cols+=("$column")
        fi
    done

    if [[ ${#label_cols[@]} -eq 0 ]]; then
        echo "Warning: no label columns found in $csv_path" >&2
        continue
    fi

    for label_col in "${label_cols[@]}"; do
        suffix="${label_col%_label}"

        if [[ "$suffix" == "true" ]]; then
            if [[ $trained_true_label -eq 1 ]]; then
                echo "Skipping true_label for $csv_path: RBM_labels_all_true already trained."
                continue
            fi
            output_model_dir="$MODEL_DIR/RBM_labels_all_true"
            trained_true_label=1
        else
            output_model_dir="$MODEL_DIR/RBM_labels_${ntrain}_${suffix}"
        fi

        cmd=(
            "$ANNADCA_BIN" train
            -d "$csv_path"
            -o "$output_model_dir"
            --column_sequences "$SEQUENCE_COLUMN"
            --column_labels "$label_col"
            --column_name "$NAME_COLUMN"
            --nepochs "$NEPOCHS"
            "${EXTRA_ARGS[@]}"
        )

        echo "Running: ${cmd[*]}"
        if [[ $DRY_RUN -eq 0 ]]; then
            "${cmd[@]}"
        fi
        num_runs=$((num_runs + 1))
    done
done

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete. Planned trainings: $num_runs"
else
    echo "Training complete. Executed trainings: $num_runs"
fi
