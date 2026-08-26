#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Regenerate the rebuttal annotation analysis for RR, t1=0.4, seeds 1-10.

Only the ESM2 and ESM-C split embeddings and their downstream predictions are
regenerated. The final classification plots include all annotation methods.

Usage:
  src/pipeline/run_rebuttal_annotation_analysis.sh \
    [--outputs_root <path>] \
    [--output_dir <path>] \
    [--esm2_model <model-id>] \
    [--esmc_model <model-id>] \
    [--use_latex] \
    [--dry_run]
EOF
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

OUTPUTS_ROOT="outputs"
ANALYSIS_OUTPUT_DIR="images/rebuttal_analysis/annotation"
ESM2_MODEL="facebook/esm2_t33_650M_UR50D"
ESMC_MODEL="biohub/ESMC-600M"
USE_LATEX=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --outputs_root) OUTPUTS_ROOT="$2"; shift 2 ;;
        --output_dir) ANALYSIS_OUTPUT_DIR="$2"; shift 2 ;;
        --esm2_model) ESM2_MODEL="$2"; shift 2 ;;
        --esmc_model) ESMC_MODEL="$2"; shift 2 ;;
        --use_latex) USE_LATEX=1; shift ;;
        --dry_run) DRY_RUN=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown argument: $1" >&2; usage; exit 1 ;;
    esac
done

readonly DATASET="RR"
readonly T1="0.4"
readonly ROC_TRAIN_SIZE="1000"
readonly -a SEEDS=(1 2 3 4 5 6 7 8 9 10)
readonly -a TRAIN_SIZES=(100 500 1000 2000 5000)
readonly -a METHODS=(onehot rbm esm2 esmc esm2_supervised)

run_command() {
    printf 'Running:'
    printf ' %q' "$@"
    printf '\n'
    if [[ $DRY_RUN -eq 0 ]]; then
        "$@"
    fi
}

for seed in "${SEEDS[@]}"; do
    case_dir="$OUTPUTS_ROOT/$DATASET/t1_$T1/seed_$seed"
    splits_dir="$case_dir/splits"
    embeddings_dir="$case_dir/embeddings"
    predictions_dir="$case_dir/predictions"

    if [[ ! -d "$splits_dir" ]]; then
        echo "Missing precomputed splits for seed $seed: $splits_dir" >&2
        exit 1
    fi
    for split_file in test.csv train_100.csv train_500.csv train_1000.csv train_2000.csv train_5000.csv; do
        if [[ ! -f "$splits_dir/$split_file" ]]; then
            echo "Missing precomputed split for seed $seed: $splits_dir/$split_file" >&2
            exit 1
        fi
    done

    run_command src/pipeline/stages/02_embed_splits.sh \
        --splits_dir "$splits_dir" \
        --output_dir "$embeddings_dir" \
        --models esmc esm2 \
        --esm2_model "$ESM2_MODEL" \
        --esmc_model "$ESMC_MODEL"

    run_command src/pipeline/stages/03_predict_test_embeddings.sh \
        --embeddings_dir "$embeddings_dir" \
        --predictions_dir "$predictions_dir" \
        --models esmc esm2
done

analysis_command=(
    python analysis/analyze_classification.py
    --dataset "$DATASET"
    --t1 "$T1"
    --seeds "${SEEDS[@]}"
    --train-sizes "${TRAIN_SIZES[@]}"
    --methods "${METHODS[@]}"
    --roc-train-size "$ROC_TRAIN_SIZE"
    --output-dir "$ANALYSIS_OUTPUT_DIR"
)
if [[ $USE_LATEX -eq 1 ]]; then
    analysis_command+=(--use-latex)
fi
run_command "${analysis_command[@]}"

if [[ $DRY_RUN -eq 1 ]]; then
    echo "Dry run complete; no embeddings, predictions, or plots were written."
else
    echo "Rebuttal annotation analysis complete."
    echo "Plots and metric tables: $ANALYSIS_OUTPUT_DIR"
fi
