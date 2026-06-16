import argparse
import json
import os
from contextlib import nullcontext
from typing import Any, Dict, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from transformers import AutoModel, AutoTokenizer

from utils import load_query_data

os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Predict labels using a supervised PLM checkpoint (frozen backbone finetuning)."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the output directory produced by train_supervised_freeze.py.",
    )
    parser.add_argument("--backbone", type=str, default=None, help="Override backbone model identifier.")
    parser.add_argument(
        "--train_csv",
        type=str,
        default=None,
        help="Optional training CSV. When provided, the output includes a train group compatible with prepare_rbm_dataset.py.",
    )
    parser.add_argument("--query", type=str, required=True, help="Path to the query dataset in .csv format.")
    parser.add_argument(
        "--output",
        type=str,
        default="predictions.plm_supervised.h5",
        help="Output .h5 file with predictions.",
    )
    parser.add_argument(
        "--info",
        type=str,
        default="",
        help="Optional metadata string saved at the top level of the output .h5 file.",
    )
    parser.add_argument("--column_sequences", type=str, default="sequence", help="CSV sequence column.")
    parser.add_argument("--column_labels", type=str, default="label", help="CSV label column.")
    parser.add_argument("--column_headers", type=str, default="header", help="CSV header column.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokenized sequence length.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for prediction.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 mixed precision (CUDA only).")
    return parser


class PredictionHead(nn.Module):
    def __init__(self, hidden_size: int, num_labels: int):
        super().__init__()
        self.classifier = nn.Linear(hidden_size, num_labels, bias=True)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        return self.classifier(pooled)


def _normalize_labels(y: np.ndarray) -> np.ndarray:
    if y.dtype.kind == "S":
        return y.astype("U")
    if y.dtype.kind == "O":
        flat = y.reshape(-1)
        if all(isinstance(v, (bytes, bytearray)) for v in flat):
            return np.array([bytes(v).decode("utf-8", errors="replace") for v in flat], dtype="U").reshape(y.shape)
    return y


def _load_label_info(model_dir: str) -> Dict[str, Any]:
    label_path = os.path.join(model_dir, "label_mapping.json")
    if not os.path.isfile(label_path):
        raise FileNotFoundError(f"Missing label mapping file: {label_path}")
    with open(label_path, "r", encoding="utf-8") as f:
        label_info = json.load(f)
    return label_info


def _load_classifier_head(model_dir: str, hidden_size: int, num_labels: int) -> PredictionHead:
    head_path = os.path.join(model_dir, "classifier_head.pt")
    if not os.path.isfile(head_path):
        raise FileNotFoundError(f"Missing classifier head file: {head_path}")
    head = PredictionHead(hidden_size=hidden_size, num_labels=num_labels)
    state = torch.load(head_path, map_location="cpu")
    head.classifier.load_state_dict(state)
    return head


def _tokenize_sequences(batch, max_length, tokenizer):
    return tokenizer.batch_encode_plus(
        batch,
        max_length=max_length,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
    )


def _predict(
    model,
    head,
    sequences,
    tokenizer,
    device,
    batch_size=32,
    max_length=256,
    use_bf16=False,
) -> Tuple[np.ndarray, np.ndarray]:
    all_logits = []
    use_autocast = use_bf16 and device.type == "cuda"
    for i in trange(0, len(sequences), batch_size, desc="Predicting"):
        batch = sequences[i : i + batch_size]
        tokenized = _tokenize_sequences(batch, max_length, tokenizer)
        input_ids = tokenized["input_ids"].to(device)
        attention_mask = tokenized["attention_mask"].to(device)
        with torch.no_grad():
            autocast_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_autocast else nullcontext()
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
                token_embeddings = outputs.last_hidden_state[:, 1:, :]
                token_mask = attention_mask[:, 1:].unsqueeze(-1).to(token_embeddings.dtype)
                pooled = (token_embeddings * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1e-9)
                logits = head(pooled)
        all_logits.append(logits.cpu())

    logits_full = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits_full, dim=-1)
    logits_full = logits_full.float()
    probs = probs.float()
    return logits_full.numpy(), probs.numpy()


def main(config: Dict[str, Any]) -> None:
    model_dir = config["model_dir"].strip()
    if not os.path.isdir(model_dir):
        raise ValueError(f"Invalid model_dir path: '{model_dir}'. Expected a directory.")

    label_info = _load_label_info(model_dir)

    backbone = config.get("backbone") or label_info.get("backbone")
    if not backbone:
        raise ValueError("Backbone model identifier not found. Provide --backbone or ensure label_mapping.json includes it.")

    label2id = {str(k): int(v) for k, v in label_info.get("label2id", {}).items()}
    id2label_raw = label_info.get("id2label", {})
    id2label = {int(k): v for k, v in id2label_raw.items()}
    if not id2label and label2id:
        id2label = {v: k for k, v in label2id.items()}
    num_labels = len(id2label) if id2label else len(label2id)
    if num_labels == 0:
        raise ValueError("label_mapping.json does not contain a valid label mapping.")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    if config.get("bf16", False) and device.type != "cuda":
        print("Warning: --bf16 requested but CUDA is not available; running in full precision.")
    elif config.get("bf16", False) and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        print("Warning: --bf16 requested but not supported on this GPU; running in full precision.")

    use_bf16 = bool(config.get("bf16", False) and device.type == "cuda" and torch.cuda.is_bf16_supported())
    if use_bf16:
        print("Using bf16 autocast for prediction.")

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(backbone, do_lower_case=False)

    print("Loading backbone model...")
    model = AutoModel.from_pretrained(model_dir)

    hidden_size = int(model.config.hidden_size)
    head = _load_classifier_head(model_dir, hidden_size=hidden_size, num_labels=num_labels)

    model = model.to(device)
    head = head.to(device)
    model.eval()
    head.eval()

    print("Loading the query dataset...")
    seq_query, headers_query, labels_query = load_query_data(
        csv_file=config["query"],
        column_sequences=config["column_sequences"],
        column_headers=config["column_headers"],
        column_labels=config["column_labels"],
    )

    train_headers = None
    train_labels = None
    if config.get("train_csv"):
        print("Loading the training dataset metadata...")
        _, train_headers, train_labels = load_query_data(
            csv_file=config["train_csv"],
            column_sequences=config["column_sequences"],
            column_headers=config["column_headers"],
            column_labels=config["column_labels"],
        )

    print("Running predictions...")
    logits, probs = _predict(
        model,
        head,
        seq_query,
        tokenizer,
        device,
        batch_size=config["batch_size"],
        max_length=config["max_length"],
        use_bf16=use_bf16,
    )

    pred_ids = logits.argmax(axis=-1)
    pred_labels = np.array([id2label.get(int(i), str(int(i))) for i in pred_ids])

    if labels_query is not None:
        labels_array = _normalize_labels(np.asarray(labels_query))
        pred_labels_norm = _normalize_labels(np.asarray(pred_labels))
        accuracy = float(np.mean(pred_labels_norm == labels_array))
        print(f"Prediction accuracy: {accuracy:.4f}")

    output_path = config["output"]
    if output_path.split(".")[-1].lower() != "h5":
        output_path += ".h5"

    print(f"Saving predictions to {output_path}...")
    with h5py.File(output_path, "w") as f:
        f.create_dataset("info", data=np.asarray(config.get("info", ""), dtype="S"))
        if train_headers is not None:
            train_group = f.create_group("train")
            train_group.create_dataset("headers", data=np.asarray(train_headers).astype("S"))
            if train_labels is not None:
                train_labels_array = _normalize_labels(np.asarray(train_labels))
                if train_labels_array.dtype.kind in ("U", "O"):
                    train_labels_array = train_labels_array.astype("S")
                train_group.create_dataset("labels_true", data=train_labels_array)

        test_group = f.create_group("test")
        test_group.create_dataset("headers", data=np.asarray(headers_query).astype("S"))

        if labels_query is not None:
            labels_array = _normalize_labels(np.asarray(labels_query))
            if labels_array.dtype.kind in ("U", "O"):
                labels_array = labels_array.astype("S")
            test_group.create_dataset("labels_true", data=labels_array)

        preds_group = test_group.create_group("predictions/plm_supervised")
        if pred_labels.dtype.kind in ("U", "O"):
            pred_labels = pred_labels.astype("S")
        preds_group.create_dataset("labels_pred", data=pred_labels)
        preds_group.create_dataset("labels_probs", data=probs)

    print("Done.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)
