#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Run all configured DELIGHT pipeline cases.

Usage:
  src/pipeline/run_experiments.sh \
    [--outputs_root <path>] \
    [--models_root <path>] \
    [config ...]

If no config paths are provided, all files under src/pipeline/config/*.conf are
used. Config files are bash fragments defining DATASET_NAME, SOURCE_CSV,
RBM_MODEL_PATH, SEEDS, T1_VALUES, and optional supervised PLM settings.
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUTS_ROOT="outputs"
MODELS_ROOT="models"

config_files=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --outputs_root) OUTPUTS_ROOT="$2"; shift 2 ;;
        --models_root) MODELS_ROOT="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) config_files+=("$1"); shift ;;
    esac
done

if [[ ${#config_files[@]} -eq 0 ]]; then
    shopt -s nullglob
    config_files=("$SCRIPT_DIR"/config/*.conf)
    shopt -u nullglob
fi

if [[ ${#config_files[@]} -eq 0 ]]; then
    echo "No pipeline config files found." >&2
    usage
    exit 1
fi

for config_file in "${config_files[@]}"; do
    if [[ ! -f "$config_file" ]]; then
        echo "Missing config file: $config_file" >&2
        exit 1
    fi

    unset DATASET_NAME SOURCE_CSV RBM_MODEL_PATH SEEDS T1_VALUES SUPERVISED_PLM_ENABLED SUPERVISED_PLM_MIN_NTRAIN
    # shellcheck source=/dev/null
    source "$config_file"

    : "${DATASET_NAME:?DATASET_NAME is required in $config_file}"
    : "${SOURCE_CSV:?SOURCE_CSV is required in $config_file}"
    : "${RBM_MODEL_PATH:?RBM_MODEL_PATH is required in $config_file}"
    : "${SEEDS:?SEEDS is required in $config_file}"
    : "${T1_VALUES:?T1_VALUES is required in $config_file}"

    supervised_plm_enabled="${SUPERVISED_PLM_ENABLED:-0}"
    supervised_plm_min_ntrain="${SUPERVISED_PLM_MIN_NTRAIN:-1000}"

    for seed in "${SEEDS[@]}"; do
        for t1 in "${T1_VALUES[@]}"; do
            src/pipeline/run_case.sh \
                --source_csv "$SOURCE_CSV" \
                --seed "$seed" \
                --dataset "$DATASET_NAME" \
                --t1 "$t1" \
                --rbm_model_path "$RBM_MODEL_PATH" \
                --outputs_root "$OUTPUTS_ROOT" \
                --models_root "$MODELS_ROOT" \
                --supervised_plm_enabled "$supervised_plm_enabled" \
                --supervised_plm_min_ntrain "$supervised_plm_min_ntrain"
        done
    done
done
