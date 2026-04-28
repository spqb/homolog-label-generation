#!/usr/bin/env bash

set -euo pipefail

run_delight_script="src/scripts/run_delight.sh"

if [[ ! -f "$run_delight_script" ]]; then
    echo "Missing script: $run_delight_script" >&2
    exit 1
fi

run_case() {
    local source_csv="$1"
    local seed="$2"
    local dataset_name="$3"
    local t1="$4"
    local rbm_model_path="$5"

    if [[ ! -f "$source_csv" ]]; then
        echo "Missing source_csv: $source_csv" >&2
        exit 1
    fi

    if [[ ! -f "$rbm_model_path" ]]; then
        echo "Missing rbm_model_path: $rbm_model_path" >&2
        exit 1
    fi

    "$run_delight_script" \
        --source_csv "$source_csv" \
        --seed "$seed" \
        --dataset_name "$dataset_name" \
        --t1 "$t1" \
        --rbm_model_path "$rbm_model_path"
}

# cm: seed 1, t1 in {0.4, 0.7}
for t1 in 0.4 0.7; do
    run_case "data/cm/cm_tested_seqs.csv" "1" "cm" "$t1" "models/cm/embedding_rbm.h5"
    
    done

# Globin: seed 1, t1 = 0.7
run_case "data/Globin/Globin_morkos.csv" "1" "Globin" "0.7" "models/Globin/embedding_rbm.h5"

# SH3: seed 1, t1 = 0.7
run_case "data/SH3/SH3.csv" "1" "SH3" "0.7" "models/SH3/embedding_rbm.h5"

# RR: seeds 1..10, t1 in {0.4, 0.7}
for seed in $(seq 1 10); do
    for t1 in 0.4 0.7; do
        run_case "data/RR/RR.csv" "$seed" "RR" "$t1" "models/RR/embedding_rbm_ptt.h5"
    done
done