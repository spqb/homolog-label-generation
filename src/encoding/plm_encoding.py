import argparse
import torch
import os
import sys
from contextlib import nullcontext
from pathlib import Path
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
from plm.pooling import boundary_special_token_ids, residue_mean_pool

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes sequences using a pre-trained protein language model.")
    parser.add_argument("--model", type=str, default="facebook/esm2_t33_650M_UR50D", help="Model identifier.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional path to a fine-tuned backbone directory.")
    parser.add_argument("--query", type=str, default=None, help="Path to the query dataset in .csv or fasta format.")
    parser.add_argument("--output", type=str, default="embeddings.esm2.h5", help="Output file containing the query sequences embeddings.")
    parser.add_argument("--info", type=str, default="", help="Optional metadata string saved at the top level of the output .h5 file.")
    parser.add_argument("--column_sequences", type=str, default="sequence", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding.")
    parser.add_argument("--bf16", action="store_true", help="Enable bfloat16 mixed precision for embedding computation (CUDA only).")
    
    return parser


def tokenize_sequences(batch, max_length, tokenizer):
    batch = [str(sequence) for sequence in batch]
    feat = tokenizer(
        batch,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )
    if "input_ids" not in feat and "sequence_tokens" in feat:
        feat["input_ids"] = feat.pop("sequence_tokens")
    return feat


def load_esmc_model_and_tokenizer(model_id: str) -> tuple[Any, Any]:
    """Load ESM-C through EvolutionaryScale's official implementation."""
    try:
        from esm.models.esmc import EsmcForMaskedLM, EsmcTokenizer
    except (ImportError, AttributeError):
        try:
            from esm.models.esmc import ESMC

            model_name = "esmc_300m" if "300m" in model_id.lower() else "esmc_600m"
            try:
                model = ESMC.from_pretrained(model_name, device=torch.device("cpu"))
            except ValueError as error:
                if "does not contain a valid checkpoint" not in str(error):
                    raise
                model = load_legacy_esmc_checkpoint(ESMC, model_name)
            return model, model.tokenizer
        except (ImportError, AttributeError) as error:
            raise RuntimeError(
                "ESM-C requires the EvolutionaryScale 'esm' package. If 'fair-esm' "
                "is also installed, uninstall it and reinstall 'esm' because both "
                "packages use the same Python namespace."
            ) from error
    else:
        tokenizer = EsmcTokenizer()
        model = EsmcForMaskedLM.from_pretrained(model_id)
        return model, tokenizer


def load_legacy_esmc_checkpoint(esmc_class: Any, model_name: str) -> Any:
    """Load old ESM-C snapshots whose weights live below data/weights/."""
    from accelerate import init_empty_weights
    from esm.tokenization import get_esmc_model_tokenizers
    from esm.utils.constants.esm3 import data_root

    architectures = {
        "esmc_300m": (960, 15, 30, "esmc-300"),
        "esmc_600m": (1152, 18, 36, "esmc-600"),
    }
    if model_name not in architectures:
        raise ValueError(f"Unsupported legacy ESM-C model: {model_name}")

    d_model, n_heads, n_layers, data_name = architectures[model_name]
    checkpoint_root = Path(data_root(data_name))
    weight_files = sorted(checkpoint_root.glob("data/weights/*.pth"))
    if len(weight_files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one ESM-C .pth file below {checkpoint_root / 'data/weights'}, "
            f"found {len(weight_files)}."
        )

    print(f"Loading legacy ESM-C weights from: {weight_files[0]}")
    with init_empty_weights():
        model = esmc_class(
            d_model=d_model,
            n_heads=n_heads,
            n_layers=n_layers,
            tokenizer=get_esmc_model_tokenizers(),
            use_flash_attn=True,
        ).eval()
    state_dict = torch.load(weight_files[0], map_location="cpu")
    model.load_state_dict(state_dict, assign=True)
    return model.eval()


def model_last_hidden_state(model_output: Any) -> torch.Tensor:
    if getattr(model_output, "last_hidden_state", None) is not None:
        return model_output.last_hidden_state
    if getattr(model_output, "embeddings", None) is not None:
        return model_output.embeddings
    raise ValueError("The protein language model did not return residue embeddings.")
        
        
def compute_embeddings(model, sequences, tokenizer, device, batch_size=32, max_length=256, use_bf16=False):
    all_embeddings = []
    special_token_ids = boundary_special_token_ids(tokenizer)
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
                if model.__class__.__name__ == "ESMC":
                    model_output = model(sequence_tokens=input_ids, sequence_id=attention_mask.bool())
                else:
                    model_output = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = residue_mean_pool(
            model_last_hidden_state(model_output),
            input_ids,
            attention_mask,
            special_token_ids,
        )
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

    checkpoint = config.get("checkpoint")
    if checkpoint is not None:
        checkpoint = checkpoint.strip()
    is_esmc = "esmc" in config["model"].lower()
    if is_esmc and checkpoint:
        raise ValueError("Fine-tuned --checkpoint loading is not supported for ESM-C.")

    if is_esmc:
        print("Loading ESM-C tokenizer and model through the official esm package...")
        model, tokenizer = load_esmc_model_and_tokenizer(config["model"])
    elif checkpoint:
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["model"], do_lower_case=False)
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
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(config["model"], do_lower_case=False)
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
