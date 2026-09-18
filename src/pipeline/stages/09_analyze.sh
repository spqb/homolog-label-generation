#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

kind="${1:-}"
[[ $# -gt 0 ]] && shift
case "$kind" in
    splits) exec python src/analysis/analyze_splits.py "$@" ;;
    classification) exec python src/analysis/analyze_classification.py "$@" ;;
    generation|generation-pca) exec python src/analysis/analyze_generated_samples.py "$@" ;;
    missing-labels) exec python src/analysis/missing_labels/analyze.py "$@" ;;
    missing-labels-summary) exec python src/analysis/missing_labels/plot_summary.py "$@" ;;
    *) echo "Usage: $0 {splits|classification|generation|generation-pca|missing-labels|missing-labels-summary} [analysis options]" >&2; exit 2 ;;
esac
