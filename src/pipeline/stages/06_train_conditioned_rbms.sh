#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/06_train_conditioned_rbms.sh \
    --datasets_dir <path> \
    --model_dir <path> \
    [--annadca_bin annadca] \
    [--methods <true|onlytrain|onehot|rbm|esm2|esmc|ankh> ...] \
    [--train_sizes <N> ...] \
    [--skip_existing] \
    [--dry_run]

Trains one conditioned RBM for each supported label column in each
rbm_dataset_ntrain_*.csv.
EOF
}

DATASETS_DIR=""
MODEL_DIR=""
ANNADCA_BIN="annadca"
DRY_RUN=0
SKIP_EXISTING=0
METHODS=()
TRAIN_SIZES=()
SEQUENCE_COLUMN="sequence_align"
NAME_COLUMN="header"
NEPOCHS="50000"
EXTRA_ARGS=(--hidden 500 --no_reweighting --lr 0.005)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets_dir) DATASETS_DIR="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --annadca_bin) ANNADCA_BIN="$2"; shift 2 ;;
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

# The ground-truth benchmark is independent of annotation budget and must use
# the fullest available train--test union rather than the first (smallest) CSV.
largest_dataset="${csv_files[-1]}"

trained_true_label=0
num_runs=0
num_skipped=0

contains() {
    local wanted="$1"; shift
    local value
    for value in "$@"; do [[ "$value" == "$wanted" ]] && return 0; done
    return 1
}

for csv_path in "${csv_files[@]}"; do
    filename="$(basename "$csv_path")"
    if [[ "$filename" =~ rbm_dataset_ntrain_([0-9]+)\.csv$ ]]; then
        ntrain="${BASH_REMATCH[1]}"
    else
        echo "Skipping unexpected filename format: $csv_path" >&2
        continue
    fi

    header_line="$(head -n 1 "$csv_path")"
    # CSV writers commonly use CRLF; do not retain CR in the last column name.
    header_line="${header_line%$'\r'}"
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
        if [[ "$label_col" == "train_only" ]]; then
            suffix="onlytrain"
        else
            suffix="${label_col%_label}"
        fi

        if [[ ${#METHODS[@]} -gt 0 ]] && ! contains "$suffix" "${METHODS[@]}"; then
            continue
        fi

        if [[ "$suffix" == "true" ]]; then
            if [[ "$csv_path" != "$largest_dataset" ]]; then
                continue
            fi
            if [[ $trained_true_label -eq 1 ]]; then
                echo "Skipping true_label for $csv_path: full-union True RBM already trained."
                continue
            fi
            output_model_dir="$MODEL_DIR/RBM_labels_all_true_full"
            trained_true_label=1
        else
            if [[ ${#TRAIN_SIZES[@]} -gt 0 ]] && ! contains "$ntrain" "${TRAIN_SIZES[@]}"; then
                continue
            fi
            output_model_dir="$MODEL_DIR/RBM_labels_${ntrain}_${suffix}"
        fi

        if [[ $SKIP_EXISTING -eq 1 && -s "$output_model_dir/params.h5" ]]; then
            echo "Skipping completed training: $output_model_dir/params.h5"
            num_skipped=$((num_skipped + 1))
            continue
        fi
        if [[ -d "$output_model_dir" && ! -s "$output_model_dir/params.h5" ]]; then
            echo "Incomplete model directory exists: $output_model_dir" >&2
            echo "Remove it or finish the interrupted training before rerunning." >&2
            exit 1
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
    echo "Dry run complete. Planned trainings: $num_runs; completed artifacts skipped: $num_skipped"
else
    echo "Training complete. Executed trainings: $num_runs; completed artifacts skipped: $num_skipped"
fi
