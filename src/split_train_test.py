#!/usr/bin/env python3
import pandas as pd
import os
import torch
from adabmDCA.fasta import get_tokens, encode_sequence
from adabmDCA.cobalt import run_cobalt
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare training and test splits.")
    parser.add_argument("--seed", type=int, default=1, help="Random seed.")
    parser.add_argument("--t1", type=float, default=0.4, help="Cobalt T1 threshold.")
    parser.add_argument("--t2", type=float, default=1.0, help="Cobalt T2 threshold.")
    parser.add_argument("--t3", type=float, default=0.7, help="Cobalt T3 threshold.")
    parser.add_argument("--num_samples_extraction", nargs='+', type=int, default=[2000, 1000, 500, 100], help="List of total number of samples to extract for training subsets.")
    parser.add_argument("--source_csv", type=str, help="Path to the source CSV file containing the dataset.")
    parser.add_argument("--output_dir", type=str, help="Directory to save the prepared splits.")
    return parser.parse_args()

args = parse_args()

T1 = args.t1
T2 = args.t2
T3 = args.t3
NUMS_EXTRACTION_SAMPLES = args.num_samples_extraction

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
tokens = get_tokens("protein")

fname_source_csv = args.source_csv
df_source = pd.read_csv(fname_source_csv)

dirname = args.output_dir
os.makedirs(dirname, exist_ok=True)

labels = df_source['label'].unique().tolist()
rng = torch.Generator(device=device).manual_seed(args.seed)
train_pool = pd.DataFrame()
test_pool = pd.DataFrame()
for label in labels:
    print(f"--> Processing label: {label}")
    df_label = df_source[df_source['label'] == label].reset_index(drop=True)
    headers = df_label['header'].values
    sequences = df_label['sequence_align'].values
    sequences_enc = encode_sequence(sequences, tokens)
    sequences_enc = torch.tensor(sequences_enc, device=device, dtype=dtype)
    
    # NOTE: Cobalt returns the biggest set as training set. For our purpose, we invert train and test sets and apply the filtering on the test set using T3 instead of T2.
    test_headers, _, train_headers,_ = run_cobalt(
        headers=headers,
        X=sequences_enc,
        t1=T1,
        t2=T2,
        t3=T3,
        max_train=None,
        max_test=None,
        rnd_gen=rng,
    )
    print(f"----> Train samples: {len(train_headers)}, Test samples: {len(test_headers)}")
    
    train_df = df_label[df_label['header'].isin(train_headers)].reset_index(drop=True)
    test_df = df_label[df_label['header'].isin(test_headers)].reset_index(drop=True)
    test_pool = pd.concat([test_pool, test_df], ignore_index=True)
    train_pool = pd.concat([train_pool, train_df], ignore_index=True)
    
# extract training subsets
for num_samples_tot in NUMS_EXTRACTION_SAMPLES:
    train_pool_subset = pd.DataFrame()
    for label in labels:
        num_samples_label = num_samples_tot // len(labels)
        train_pool_label = train_pool[train_pool['label'] == label].reset_index(drop=True)
        if len(train_pool_label) <= num_samples_label:
            train_pool_subset = pd.concat([train_pool_subset, train_pool_label], ignore_index=True)
        else:
            train_pool_label_sampled = train_pool_label.sample(n=num_samples_label, random_state=args.seed, replace=False).reset_index(drop=True)
            train_pool_subset = pd.concat([train_pool_subset, train_pool_label_sampled], ignore_index=True)
    train_pool_subset.to_csv(os.path.join(dirname, f"train_{num_samples_tot}.csv"), index=False)
            
# extract test subsets. take the same number of samples for each label as the smallest class in the test pool
test_pool_subset = pd.DataFrame()
min_test_samples = test_pool['label'].value_counts().min()
for label in labels:
    test_pool_label = test_pool[test_pool['label'] == label].reset_index(drop=True)
    if len(test_pool_label) <= min_test_samples:
        test_pool_subset = pd.concat([test_pool_subset, test_pool_label], ignore_index=True)
    else:
        test_pool_label_sampled = test_pool_label.sample(n=min_test_samples, random_state=args.seed, replace=False).reset_index(drop=True)
        test_pool_subset = pd.concat([test_pool_subset, test_pool_label_sampled], ignore_index=True)

test_pool_subset.to_csv(os.path.join(dirname, f"test.csv"), index=False)
