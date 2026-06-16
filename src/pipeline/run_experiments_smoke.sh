#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

src/pipeline/run_experiments.sh \
    --outputs_root tmp/smoke_outputs \
    --models_root tmp/smoke_models \
    src/pipeline/config/smoke_rr.conf
