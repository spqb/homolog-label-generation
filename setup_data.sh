#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: ./setup_data.sh [ARCHIVE] [--force]

Create the local data/ and models/ directories from the Zenodo archive.

Arguments:
  ARCHIVE   Path to Data_delight.zip. Defaults to ./Data_delight.zip.
  --force   Replace existing data/ and models/ directories.
EOF
}

ARCHIVE="Data_delight.zip"
FORCE=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            usage
            exit 0
            ;;
        --force)
            FORCE=1
            shift
            ;;
        -*)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 1
            ;;
        *)
            ARCHIVE="$1"
            shift
            ;;
    esac
done

if [[ ! -f "$ARCHIVE" ]]; then
    echo "Archive not found: $ARCHIVE" >&2
    echo "Download it from: https://zenodo.org/records/20719564/files/Data_delight.zip" >&2
    exit 1
fi

if ! command -v unzip >/dev/null 2>&1; then
    echo "Missing required command: unzip" >&2
    exit 1
fi

TMPDIR="$(mktemp -d)"
cleanup() {
    rm -rf "$TMPDIR"
}
trap cleanup EXIT

unzip -q "$ARCHIVE" -d "$TMPDIR"

EXTRACTED_ROOT="$TMPDIR/Data_delight"
if [[ ! -d "$EXTRACTED_ROOT/data" || ! -d "$EXTRACTED_ROOT/rbm_models" ]]; then
    echo "Unexpected archive layout. Expected Data_delight/data and Data_delight/rbm_models." >&2
    exit 1
fi

for target in data models; do
    if [[ -e "$target" && "$FORCE" -ne 1 ]]; then
        echo "Refusing to overwrite existing $target/." >&2
        echo "Move it aside or rerun with --force." >&2
        exit 1
    fi
done

if [[ "$FORCE" -eq 1 ]]; then
    rm -rf data models
fi

mkdir -p data models
cp -R "$EXTRACTED_ROOT/data/." data/
cp -R "$EXTRACTED_ROOT/rbm_models/." models/

echo "Created data/ and models/ from $ARCHIVE."
