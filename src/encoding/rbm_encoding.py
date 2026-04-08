import argparse
import numpy as np
import torch
import os
import sys
import h5py

src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)
    
from adabmDCA.fasta import get_tokens, encode_sequence
from rbms.utils import get_saved_updates
from rbms.io import load_params
from utils import load_query_data


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes aligned sequences using RBM embedding and saves them as HDF5 archives.")
    parser.add_argument("--model", type=str, required=True, help="Path to the RBM model file.")
    parser.add_argument("--query", type=str, required=True, help="Path to the input dataset in .csv format.")
    parser.add_argument("--output", type=str, default=None, help="Output file containing the RBM encoded sequences. If not provided, it will be saved with the same name as the input file but with .rbm.h5 extension.")
    parser.add_argument("--info", type=str, default="", help="Optional metadata string saved at the top level of the output .h5 file.")
    parser.add_argument("--column_sequences", type=str, default="sequence_align", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    
    return parser


def rbm_encode_sequences(sequences, rbm_params, tokens, device, dtype):
    """
    RBM encode a list of aligned sequences.
    
    Args:
        sequences: List of aligned protein sequences (all same length)
        rbm_params: Loaded RBM model parameters
        tokens: Token list from adabmDCA
        device: PyTorch device
        dtype: PyTorch dtype
    
    Returns:
        RBM encoded array of shape (n_sequences, n_hidden)
    """
    # Encode sequences to integer representation
    encoded = encode_sequence(sequences, tokens)
    
    # Convert to torch tensor
    encoded_tensor = torch.tensor(encoded).to(dtype=dtype, device=device)
    
    # Encode using RBM
    inputs = {"visible": encoded_tensor}
    hidden = rbm_params.sample_hiddens(inputs)["hidden_mag"].cpu().numpy()
    
    return hidden


def main(config):
    assert os.path.exists(config["query"]), f"Input file {config['query']} does not exist."
    assert os.path.exists(config["model"]), f"RBM model file {config['model']} does not exist."
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32
    print(f"Using device: {device}")
    
    # Get protein tokens
    tokens = get_tokens("protein")
    print(f"Using {len(tokens)} tokens for encoding")
    
    # Load RBM model
    print(f"Loading RBM model from {config['model']}...")
    saved_updates = get_saved_updates(filename=config["model"])
    #params, *_ = load_model(filename=config["model"], index=saved_updates[-1], device=device, dtype=dtype)
    params = load_params(filename=config["model"], index=saved_updates[-1], device=device, dtype=dtype)
    print("RBM model loaded successfully")
    
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
        labels_array = np.asarray(labels)
        print(f"Found labels with {len(np.unique(labels_array))} unique values")
    
    # RBM encode sequences
    print("RBM encoding aligned sequences...")
    rbm_embeddings = rbm_encode_sequences(sequences, params, tokens, device, dtype)
    print(f"RBM encoded shape: {rbm_embeddings.shape}")
    
    # Prepare output filename
    if config["output"] is not None:
        output_path = config["output"]
    else:
        output_prefix = os.path.splitext(config["query"])[0]
        output_path = f"{output_prefix}.rbm.h5"

    if output_path.split(".")[-1].lower() != "h5":
        output_path += ".h5"
    
    # Save to HDF5 archive
    print(f"Saving RBM encoding to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("info", data=np.asarray(config.get("info", ""), dtype="S"))
        f.create_dataset("embeddings", data=rbm_embeddings)
        f.create_dataset("headers", data=np.asarray(headers).astype("S"))
        if labels is not None:
            labels_array = np.asarray(labels)
            if labels_array.dtype.kind in ("U", "O"):
                labels_array = labels_array.astype("S")
            f.create_dataset("labels", data=labels_array)
    
    print(f"Successfully saved RBM encoding to {output_path}")
    print("Done!")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
