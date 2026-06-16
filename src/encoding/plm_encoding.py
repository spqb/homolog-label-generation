import argparse
import torch
import os
import sys
from contextlib import nullcontext
from typing import Any
import numpy as np
import h5py
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

# Add parent directory to path to import train modules
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from utils import load_query_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes sequences using a pre-trained protein language model.")
    parser.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D", help="Model identifier.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a fine-tuned backbone directory.")
    parser.add_argument("--query", type=str, default=None, help="Path to the query dataset in .csv or fasta format.")
    parser.add_argument("--output", type=str, default="embeddings.esm2_t33_650M_UR50D.h5", help="Output file containing the query sequences embeddings.")
    parser.add_argument("--info", type=str, default="", help="Optional metadata string saved at the top level of the output .h5 file.")
    parser.add_argument("--column_sequences", type=str, default="sequence", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision for embedding computation (CUDA only).")
    
    return parser


def tokenize_sequences(batch, max_length, tokenizer):
    feat = tokenizer.batch_encode_plus(
        batch, 
        max_length=max_length, 
        return_tensors='pt', 
        padding='max_length', 
        truncation=True
    )
    return feat
        
        
def compute_embeddings(model, sequences, tokenizer, device, batch_size=32, max_length=256, use_bf16=False):
    all_embeddings = []
    use_autocast = use_bf16 and device.type == "cuda"
    pbar = tqdm(total=len(sequences), leave=False)
    for i in range(0, len(sequences), batch_size):
        pbar.update(min(batch_size, len(sequences) - i))
        batch = sequences[i:i+batch_size]
        tokenized_batch = tokenize_sequences(batch, max_length, tokenizer)
        input_ids = tokenized_batch["input_ids"].to(device)
        attention_mask = tokenized_batch["attention_mask"].to(device)
        with torch.no_grad():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
            with autocast_ctx:
                model_output = model(input_ids=input_ids, attention_mask=attention_mask)
        # Remove CLS token (position 0), then mean-pool non-padding tokens
        token_embeddings = model_output.last_hidden_state[:, 1:, :]
        token_mask = attention_mask[:, 1:].unsqueeze(-1).to(token_embeddings.dtype)
        embeddings = torch.sum(token_embeddings * token_mask, 1) / torch.sum(token_mask, 1)
        all_embeddings.append(embeddings.cpu())
    pbar.close()
    
    return torch.cat(all_embeddings, dim=0)


def main(config):
    model: Any
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    if config.get("bf16", False) and device.type != "cuda":
        print("Warning: --bf16 requested but CUDA is not available; running in full precision.")
    elif config.get("bf16", False) and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("Warning: --bf16 requested but not supported on this GPU; running in full precision.")

    use_bf16 = bool(config.get("bf16", False) and device.type == "cuda" and torch.cuda.is_bf16_supported())
    if use_bf16:
        print("Using bf16 autocast for embedding computation.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model"], do_lower_case=False)

    checkpoint = config.get("checkpoint")
    if checkpoint is not None:
        checkpoint = checkpoint.strip()
    if checkpoint:
        if not os.path.isdir(checkpoint):
            raise ValueError(f"Invalid checkpoint path: '{checkpoint}'. Expected a fine-tuned backbone directory.")

        load_path = checkpoint
        config_path = os.path.join(load_path, "config.json")
        if not os.path.isfile(config_path):
            parent_path = os.path.abspath(os.path.join(load_path, os.pardir))
            parent_config = os.path.join(parent_path, "config.json")
            if os.path.isfile(parent_config):
                load_path = parent_path

        print(f"Loading fine-tuned backbone from: {load_path}")
        try:
            model = AutoModel.from_pretrained(load_path)
        except ValueError:
            base_model_id = config.get("model")
            if not base_model_id:
                raise
            print(f"Falling back to base model '{base_model_id}' and loading weights from {load_path}.")
            model = AutoModel.from_pretrained(base_model_id)
            weights_path = os.path.join(load_path, "pytorch_model.bin")
            if not os.path.isfile(weights_path):
                raise FileNotFoundError(f"Missing weights file: {weights_path}")
            state = torch.load(weights_path, map_location="cpu")
            model.load_state_dict(state, strict=False)
    else:
        print("Loading model...")
        model = AutoModel.from_pretrained(config["model"])

    model = model.to(device)
    model.eval()
        
    print("Loading the query dataset...")
    seq_query, headers_query, labels_query = load_query_data(
        csv_file=config["query"],
        column_sequences=config["column_sequences"],
        column_headers=config["column_headers"],
        column_labels=config["column_labels"]
    )

    print("Embedding the query dataset...")
    X_test = compute_embeddings(
        model,
        seq_query,
        tokenizer,
        device,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        use_bf16=use_bf16,
    ).numpy()

    output_path = config["output"]
    if output_path.split(".")[-1].lower() != "h5":
        output_path += ".h5"
    
    print("Saving the query dataset's embeddings...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("info", data=np.asarray(config.get("info", ""), dtype="S"))
        f.create_dataset("embeddings", data=X_test)
        f.create_dataset("headers", data=np.asarray(headers_query).astype("S"))
        if labels_query is not None:
            labels_array = np.asarray(labels_query)
            if labels_array.dtype.kind in ("U", "O"):
                labels_array = labels_array.astype("S")
            f.create_dataset("labels", data=labels_array)

if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)