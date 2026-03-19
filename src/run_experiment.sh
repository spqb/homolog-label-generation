#!/bin/bash

# ============================================================
# Response Regulators
# ============================================================

# 1. Prepare splits for repeated experiments
T1=0.4
T2=1.0
T3=0.7
source_csv="./data/RR/RR.csv"
output_dir="./outputs/RR/t1_${T1}"
num_seeds=10

python src/prepare_splits_repeated.py --num_seeds $num_seeds --t1 $T1 --t2 $T2 --t3 $T3 --source_csv $source_csv --output_dir $output_dir

for seed in $(seq 0 $((num_seeds - 1))); do
    # 2. Embeddings
    # 2.1 ESM2 embeddings
    python src/pretrained_encoding.py \
        --query "${output_dir}/seed_${seed}.test.csv" \
        --output "${output_dir}/seed_${seed}.test.embedding.esm2.npz" \
        --batch_size 32
    
    for ntrain in 100 500 1000 2000; do
        # 2.2 ESM2 embeddings for training subsets
        python src/pretrained_encoding.py \
            --query "${output_dir}/seed_${seed}.train_${ntrain}.csv" \
            --output "${output_dir}/seed_${seed}.train_${ntrain}.embedding.esm2.npz" \
            --batch_size 32
    done
    # 2.3 RBM embeddings
    python src/rbm_encoding.py \
        --query "${output_dir}/seed_${seed}.test.csv" \
        --output "${output_dir}/seed_${seed}.test.embedding.rbm.npz" \
        --rbm_model_path "./models/RR/embedding_rbm_ptt.h5"
    
    for ntrain in 100 500 1000 2000; do
        # 2.4 RBM embeddings for training subsets
        python src/rbm_encoding.py \
            --query "${output_dir}/seed_${seed}.train_${ntrain}.csv" \
            --output "${output_dir}/seed_${seed}.train_${ntrain}.embedding.rbm.npz" \
            --rbm_model_path "./models/RR/embedding_rbm_ptt.h5"
    done
    # 2.5 One-hot encoding
    python src/onehot_encoding.py \
        --query "${output_dir}/seed_${seed}.test.csv" \
        --output "${output_dir}/seed_${seed}.test.embedding.onehot.npz"
    
    # 2.6 One-hot encoding for training subsets
    for ntrain in 100 500 1000 2000; do
        python src/onehot_encoding.py \
            --query "${output_dir}/seed_${seed}.train_${ntrain}.csv" \
            --output "${output_dir}/seed_${seed}.train_${ntrain}.embedding.onehot.npz"
    done

    # 3. Predictions using logistic regression
    for embedding in "esm2" "rbm" "onehot"; do
        for ntrain in 100 500 1000 2000; do
            python src/predict_from_embeddings.py \
                --train_npz "${output_dir}/seed_${seed}.train_${ntrain}.embedding.${embedding}.npz" \
                --test_npz "${output_dir}/seed_${seed}.test.embedding.${embedding}.npz" \
                --flag "ntrain_${ntrain}"
        done
    done
done