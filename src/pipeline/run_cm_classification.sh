#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
echo "Deprecated: use src/pipeline/run_experiments.sh --profile classification --datasets cm" >&2
exec bash "$SCRIPT_DIR/run_experiments.sh" --profile classification --datasets cm "$@"
