#!/usr/bin/env bash

set -euo pipefail

usage() {
	cat <<'EOF'
Automate annaDCA trainings from RBM dataset CSV files.

Usage:
	./automate_rbm_training.sh \
		--input_dir outputs/cm/t1_0.4/seed_1/datasets_for_rbm \
		--output_dir models/CM \
		[--label_columns rbm_label,onehot_label,true_label] \
		[--sequence_column sequence_align] \
		[--name_column header] \
		[--nepochs 30000] \
		[--annadca_bin annadca] \
		[--dry_run] \
		[--batch_size 2000 --learning_rate 0.05 --dtype float32]

Notes:
- By default, label columns are auto-detected from each CSV header as columns ending with '_label'.
- CSV files are discovered recursively under input_dir and filtered by pattern:
	rbm_dataset_ntrain_*.csv
- Output naming follows your RR convention:
	- <label>_label + ntrain => RBM_labels_<ntrain>_<label>
	- true_label => RBM_labels_true (trained once, first matching CSV only)
- Any argument not recognized by this wrapper is forwarded to 'annadca train'.
- '--nepochs' is forwarded only if provided; otherwise, annadca default is used.
EOF
}

INPUT_DIR=""
OUTPUT_DIR=""
LABEL_COLUMNS=""
SEQUENCE_COLUMN="sequence_align"
NAME_COLUMN="header"
NEPOCHS=""
ANNADCA_BIN="annadca"
DRY_RUN=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
	case "$1" in
		--input_dir)
			INPUT_DIR="$2"
			shift 2
			;;
		--output_dir)
			OUTPUT_DIR="$2"
			shift 2
			;;
		--label_columns)
			LABEL_COLUMNS="$2"
			shift 2
			;;
		--sequence_column)
			SEQUENCE_COLUMN="$2"
			shift 2
			;;
		--name_column)
			NAME_COLUMN="$2"
			shift 2
			;;
		--nepochs)
			NEPOCHS="$2"
			shift 2
			;;
		--annadca_bin)
			ANNADCA_BIN="$2"
			shift 2
			;;
		--dry_run)
			DRY_RUN=1
			shift
			;;
		--help|-h)
			usage
			exit 0
			;;
		--)
			shift
			while [[ $# -gt 0 ]]; do
				EXTRA_ARGS+=("$1")
				shift
			done
			break
			;;
		*)
			EXTRA_ARGS+=("$1")
			shift
			;;
	esac
done

if [[ -z "$INPUT_DIR" || -z "$OUTPUT_DIR" ]]; then
	echo "Error: --input_dir and --output_dir are required." >&2
	usage
	exit 1
fi

if [[ ! -d "$INPUT_DIR" ]]; then
	echo "Error: input_dir does not exist: $INPUT_DIR" >&2
	exit 1
fi

if [[ $DRY_RUN -eq 0 ]]; then
	if ! command -v "$ANNADCA_BIN" >/dev/null 2>&1; then
		echo "Error: annadca executable not found: $ANNADCA_BIN" >&2
		exit 1
	fi
fi

mkdir -p "$OUTPUT_DIR"

mapfile -t CSV_FILES < <(find "$INPUT_DIR" -type f -name 'rbm_dataset_ntrain_*.csv' | sort -V)

if [[ ${#CSV_FILES[@]} -eq 0 ]]; then
	echo "Error: no rbm_dataset_ntrain_*.csv files found under $INPUT_DIR" >&2
	exit 1
fi

TRAINED_TRUE_LABEL=0
N_RUNS=0

for csv_path in "${CSV_FILES[@]}"; do
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
	if [[ -n "$LABEL_COLUMNS" ]]; then
		IFS=',' read -r -a requested_cols <<< "$LABEL_COLUMNS"
		for requested in "${requested_cols[@]}"; do
			found=0
			for c in "${columns[@]}"; do
				if [[ "$c" == "$requested" ]]; then
					label_cols+=("$requested")
					found=1
					break
				fi
			done
			if [[ $found -eq 0 ]]; then
				echo "Warning: requested label column '$requested' not found in $csv_path" >&2
			fi
		done
	else
		for c in "${columns[@]}"; do
			if [[ "$c" == *_label ]]; then
				label_cols+=("$c")
			fi
		done
	fi

	if [[ ${#label_cols[@]} -eq 0 ]]; then
		echo "Warning: no label columns found in $csv_path" >&2
		continue
	fi

	for label_col in "${label_cols[@]}"; do
		suffix="${label_col%_label}"

		if [[ "$suffix" == "true" ]]; then
			if [[ $TRAINED_TRUE_LABEL -eq 1 ]]; then
				echo "Skipping true_label for $csv_path (RBM_labels_true already trained)."
				continue
			fi
			model_dir="$OUTPUT_DIR/RBM_labels_true"
			TRAINED_TRUE_LABEL=1
		else
			model_dir="$OUTPUT_DIR/RBM_labels_${ntrain}_${suffix}"
		fi

		cmd=(
			"$ANNADCA_BIN" train
			-d "$csv_path"
			-o "$model_dir"
			--column_sequences "$SEQUENCE_COLUMN"
			--column_labels "$label_col"
			--column_name "$NAME_COLUMN"
		)

		if [[ -n "$NEPOCHS" ]]; then
			cmd+=(--nepochs "$NEPOCHS")
		fi

		if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
			cmd+=("${EXTRA_ARGS[@]}")
		fi

		echo "Running: ${cmd[*]}"
		if [[ $DRY_RUN -eq 0 ]]; then
			"${cmd[@]}"
		fi
		N_RUNS=$((N_RUNS + 1))
	done
done

if [[ $DRY_RUN -eq 1 ]]; then
	echo "Dry run complete. Planned trainings: $N_RUNS"
else
	echo "Training complete. Executed trainings: $N_RUNS"
fi
