import argparse
import numpy as np
import os
import sys

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from adabmDCA.fasta import get_tokens, encode_sequence
from utils import load_query_data


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-hot encodes aligned sequences and saves them as npz files.")
    parser.add_argument("--query", type=str, required=True, help="Path to the input dataset in .csv format.")
    parser.add_argument("--output", type=str, default=None, help="Output file containing the one-hot encoded sequences. If not provided, it will be saved with the same name as the input file but with .onehot.npz extension.")
    parser.add_argument("--column_sequences", type=str, default="sequence_align", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    
    return parser


def onehot_encode_sequences(sequences, tokens):
    """
    One-hot encode a list of aligned sequences.
    
    Args:
        sequences: List of aligned protein sequences (all same length)
        tokens: Token list from adabmDCA
    
    Returns:
        One-hot encoded array of shape (n_sequences, sequence_length * n_tokens)
    """
    # Encode sequences to integer representation
    encoded = encode_sequence(sequences, tokens)
    
    # One-hot encode
    n_sequences, sequence_length = encoded.shape
    n_tokens = len(tokens)
    onehot = np.eye(n_tokens)[encoded]  # Shape: (n_sequences, sequence_length, n_tokens)
    onehot = onehot.reshape(n_sequences, sequence_length * n_tokens)  # Shape: (n_sequences, sequence_length * n_tokens)
    
    return onehot


def main(config):
    assert os.path.exists(config["query"]), f"Input file {config['query']} does not exist."
    
    # Get protein tokens
    tokens = get_tokens("protein")
    print(f"Using {len(tokens)} tokens for encoding")    
    print(f"Loading input dataset from {config['query']}...")
    
    # Load CSV file
    sequences, headers, labels = load_query_data(
        csv_file=config["query"],
        column_sequences=config["column_sequences"],
        column_headers=config["column_headers"],
        column_labels=config["column_labels"]
    )
    
    print(f"Loaded {len(sequences)} sequences from CSV file")
    if labels is not None:
        print(f"Found labels with {len(np.unique(labels))} unique values")
    
    # One-hot encode sequences
    print("One-hot encoding aligned sequences...")
    onehot_embeddings = onehot_encode_sequences(sequences, tokens)
    print(f"One-hot encoded shape: {onehot_embeddings.shape}")
    
    # Prepare output filename
    if config["output"] is not None:
        output_path = config["output"]
    else:
        output_prefix = os.path.splitext(config["query"])[0]
        output_path = f"{output_prefix}.onehot.npz"
    
    # Save to npz file
    print(f"Saving one-hot encoding to {output_path}...")
    if labels is not None:
        np.savez_compressed(
            output_path,
            embeddings=onehot_embeddings,
            labels=labels,
            headers=headers
        )
    else:
        np.savez_compressed(
            output_path,
            embeddings=onehot_embeddings,
            headers=headers
        )
    
    print(f"Successfully saved one-hot encoding to {output_path}")
    print("Done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
