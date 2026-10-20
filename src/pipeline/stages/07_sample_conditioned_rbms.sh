#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  src/pipeline/stages/07_sample_conditioned_rbms.sh \
    --datasets_dir <path> \
    --model_dir <path> \
    [--annadca_bin annadca] \
    [--samples_dir <path>] \
    [--methods <method-id> ...] \
    [--train_sizes <N> ...] \
    [--self_consistency_per_label <N; default 3000>] \
    [--gibbs_steps <N; default 50000>] \
    [--skip_existing] \
    [--dry_run]

Creates two independent sample sets for each conditioned RBM:
  samples.true_labels.csv uses true_label and preserves the natural label counts;
  samples.self_consistency.csv uses 3000 balanced targets per model label.
EOF
}

DATASETS_DIR=""
MODEL_DIR=""
SAMPLES_DIR=""
ANNADCA_BIN="annadca"
PARAMS_FILE="params.h5"
NATURAL_SAMPLES_FILE="samples.true_labels.csv"
SELF_TARGETS_FILE="targets.self_consistency.csv"
SELF_SAMPLES_FILE="samples.self_consistency.csv"
SELF_CONSISTENCY_PER_LABEL=3000
GIBBS_STEPS=50000
METHODS=()
TRAIN_SIZES=()
SKIP_EXISTING=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets_dir) DATASETS_DIR="$2"; shift 2 ;;
        --model_dir) MODEL_DIR="$2"; shift 2 ;;
        --samples_dir) SAMPLES_DIR="$2"; shift 2 ;;
        --annadca_bin) ANNADCA_BIN="$2"; shift 2 ;;
        --self_consistency_per_label) SELF_CONSISTENCY_PER_LABEL="$2"; shift 2 ;;
        --gibbs_steps) GIBBS_STEPS="$2"; shift 2 ;;
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

[[ "$SELF_CONSISTENCY_PER_LABEL" =~ ^[1-9][0-9]*$ ]] || {
    echo "--self_consistency_per_label must be a positive integer" >&2; exit 1;
}
[[ "$GIBBS_STEPS" =~ ^[1-9][0-9]*$ ]] || {
    echo "--gibbs_steps must be a positive integer" >&2; exit 1;
}

if [[ -z "$DATASETS_DIR" || -z "$MODEL_DIR" ]]; then
    echo "Missing required arguments." >&2
    usage
    exit 1
fi

if [[ $DRY_RUN -eq 0 ]] && ! command -v "$ANNADCA_BIN" >/dev/null 2>&1; then
    echo "annadca executable not found: $ANNADCA_BIN" >&2
    exit 1
fi

if [[ ! -d "$DATASETS_DIR" ]]; then
    echo "datasets_dir does not exist: $DATASETS_DIR" >&2
    exit 1
fi

if [[ ! -d "$MODEL_DIR" ]]; then
    echo "model_dir does not exist: $MODEL_DIR" >&2
    exit 1
fi

shopt -s nullglob
dataset_files=("$DATASETS_DIR"/rbm_dataset_ntrain_*.csv)
model_dirs=("$MODEL_DIR"/RBM_labels_*)
shopt -u nullglob

if [[ ${#dataset_files[@]} -eq 0 ]]; then
    echo "No rbm_dataset_ntrain_*.csv files found in: $DATASETS_DIR" >&2
    exit 1
fi

if [[ ${#model_dirs[@]} -eq 0 ]]; then
    echo "No RBM_labels_* model directories found in: $MODEL_DIR" >&2
    exit 1
fi

largest_dataset="$(
    printf '%s\n' "${dataset_files[@]}" | sort -V | tail -n 1
)"
echo "Generation Gibbs steps: $GIBBS_STEPS"

num_runs=0
num_skipped=0
contains() {
    local wanted="$1"; shift
    local value
    for value in "$@"; do [[ "$value" == "$wanted" ]] && return 0; done
    return 1
}
for model_path in "${model_dirs[@]}"; do
    [[ -d "$model_path" ]] || continue

    model_name="$(basename "$model_path")"
    dataset_path=""
    training_label_column=""

    if [[ "$model_name" == "RBM_labels_all_true_full" ]]; then
        method="true"
        ntrain=""
        dataset_path="$largest_dataset"
        training_label_column="true_label"
    elif [[ "$model_name" =~ ^RBM_labels_([0-9]+)_(.+)$ ]]; then
        ntrain="${BASH_REMATCH[1]}"
        label_suffix="${BASH_REMATCH[2]}"
        method="$label_suffix"
        dataset_path="$DATASETS_DIR/rbm_dataset_ntrain_${ntrain}.csv"
        if [[ "$label_suffix" == "onlytrain" ]]; then
            training_label_column="train_only"
        else
            training_label_column="${label_suffix}_label"
        fi
    else
        [[ "$model_name" == "RBM_labels_all_true" ]] && {
            echo "Skipping legacy True model trained before full-union separation: $model_path" >&2
            continue
        }
        echo "Skipping unexpected model directory name: $model_path" >&2
        continue
    fi

    if [[ ${#METHODS[@]} -gt 0 ]] && ! contains "$method" "${METHODS[@]}"; then
        continue
    fi
    if [[ -n "$ntrain" && ${#TRAIN_SIZES[@]} -gt 0 ]] && ! contains "$ntrain" "${TRAIN_SIZES[@]}"; then
        continue
    fi

    artifact_dir="${SAMPLES_DIR:-$model_path}"
    [[ -z "$SAMPLES_DIR" ]] || artifact_dir="$SAMPLES_DIR/$model_name"
    [[ $DRY_RUN -eq 1 ]] || mkdir -p "$artifact_dir"
    params_path="$model_path/$PARAMS_FILE"
    natural_samples_path="$artifact_dir/$NATURAL_SAMPLES_FILE"
    self_targets_path="$artifact_dir/$SELF_TARGETS_FILE"
    self_samples_path="$artifact_dir/$SELF_SAMPLES_FILE"
    natural_marker="$artifact_dir/.samples.true_labels.gibbs_${GIBBS_STEPS}.complete"
    self_marker="$artifact_dir/.samples.self_consistency.gibbs_${GIBBS_STEPS}.complete"

    if [[ ! -f "$dataset_path" ]]; then
        echo "Skipping $model_name: missing dataset $dataset_path" >&2
        continue
    fi

    if [[ ! -f "$params_path" ]]; then
        echo "Skipping $model_name: missing params $params_path" >&2
        continue
    fi

    if [[ $SKIP_EXISTING -eq 1 && -s "$natural_samples_path" && -f "$natural_marker" ]]; then
        echo "Skipping completed true-label samples: $natural_samples_path"
        num_skipped=$((num_skipped + 1))
    else
        echo "Sampling $model_name with true_label targets"
        echo "  dataset: $dataset_path"
        echo "  params : $params_path"
        echo "  output : $natural_samples_path"
        if [[ $DRY_RUN -eq 0 ]]; then
            "$ANNADCA_BIN" sample \
                -d "$dataset_path" \
                -p "$params_path" \
                -o "$natural_samples_path" \
                --column_name header \
                --column_label true_label \
                --column_sequence sequence_align \
                --gibbs_steps "$GIBBS_STEPS"
            touch "$natural_marker"
        fi
        num_runs=$((num_runs + 1))
    fi

    # Train-only has no independent predictor and is omitted from self-consistency.
    if [[ "$method" == "onlytrain" ]]; then
        continue
    fi

    if [[ $SKIP_EXISTING -eq 1 && -s "$self_samples_path" && -f "$self_marker" ]]; then
        echo "Skipping completed self-consistency samples: $self_samples_path"
        num_skipped=$((num_skipped + 1))
    else
        if [[ $DRY_RUN -eq 0 ]]; then
            python src/pipeline/prepare_self_consistency_targets.py \
                --dataset "$dataset_path" \
                --label-column "$training_label_column" \
                --per-label "$SELF_CONSISTENCY_PER_LABEL" \
                --output "$self_targets_path"
        else
            echo "Would prepare $SELF_CONSISTENCY_PER_LABEL self-consistency targets per label from $training_label_column"
        fi
        echo "Sampling $model_name for self-consistency"
        echo "  targets: $self_targets_path"
        echo "  output : $self_samples_path"
        if [[ $DRY_RUN -eq 0 ]]; then
            "$ANNADCA_BIN" sample \
                -d "$self_targets_path" \
                -p "$params_path" \
                -o "$self_samples_path" \
                --column_name header \
                --column_label target_label \
                --column_sequence sequence_align \
                --gibbs_steps "$GIBBS_STEPS"
            touch "$self_marker"
        fi
        num_runs=$((num_runs + 1))
    fi
done

if [[ $num_runs -eq 0 && $num_skipped -eq 0 ]]; then
    echo "No RBM sampling runs were executed." >&2
    exit 1
fi

echo "Completed $num_runs RBM sampling runs; completed artifacts skipped: $num_skipped."
