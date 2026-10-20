#!/usr/bin/env bash
set -euo pipefail
echo "Retired: generation is now configured for RR only. Use src/pipeline/run_experiments.sh --profile classification for Globin, SH3, and cm." >&2
exit 2
