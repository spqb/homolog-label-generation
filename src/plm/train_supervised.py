import argparse
import json
import os
import random
from typing import Dict, List, cast

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, concatenate_datasets, load_dataset
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from transformers import AutoModel, AutoTokenizer
from transformers import EarlyStoppingCallback, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding


class ESMSequenceClassifier(nn.Module):
    """
    Sequence classifier with mean pooling over token embeddings after removing CLS token.

    Head architecture:
      Dropout -> Dense -> Tanh -> Dropout -> Dense -> Softmax
    """

    def __init__(
        self,
        backbone_name: str,
        num_labels: int,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.num_labels = num_labels

        self.classifier = nn.Linear(hidden_size, num_labels, bias=True)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

        # Remove CLS token (position 0), then mean-pool non-padding tokens
        token_embeddings = outputs.last_hidden_state[:, 1:, :]
        token_mask = attention_mask[:, 1:].unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * token_mask).sum(dim=1) / token_mask.sum(dim=1).clamp(min=1e-9)

        logits = self.classifier(pooled)
        probabilities = self.softmax(logits)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)

        return {"loss": loss, "logits": logits, "probabilities": probabilities}


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised finetuning of ESM2 with frozen early layers.")
    parser.add_argument("--backbone", type=str, default="facebook/esm2_t33_650M_UR50D", help="Backbone model identifier.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV.")
    parser.add_argument("--val_csv", type=str, default=None, help="Optional path to validation CSV.")
    parser.add_argument("--test_csv", type=str, default=None, help="Optional path to test CSV for evaluation.")
    parser.add_argument("--folder_params", type=str, required=True, help="Output directory.")

    parser.add_argument("--column_sequences", type=str, default="sequence", help="CSV sequence column.")
    parser.add_argument("--column_labels", type=str, default="label", help="CSV label column.")

    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokenized sequence length.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in classification head.")
    parser.add_argument("--val_fraction", type=float, default=0.2, help="Fraction of each label class to hold out for validation.")
    parser.add_argument("--patience", type=int, default=3, help="Early stopping patience in evaluation steps.")
    parser.add_argument("--train_last_n", type=int, default=6, help="Number of last backbone layers to unfreeze.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser


def _build_label_mapping(dataset: Dataset, label_col: str) -> Dict[str, int]:
    labels = dataset[label_col]
    unique_labels: List[str] = sorted({str(x) for x in labels})
    return {label: i for i, label in enumerate(unique_labels)}


def _stratified_split(
    dataset: Dataset,
    label_col: str,
    val_fraction: float,
    seed: int,
) -> (Dataset, Dataset):
    if val_fraction <= 0:
        return dataset, cast(Dataset, dataset.select([]))

    label_to_indices: Dict[str, List[int]] = {}
    for idx, label in enumerate(dataset[label_col]):
        key = str(label)
        label_to_indices.setdefault(key, []).append(idx)

    rng = random.Random(seed)
    train_indices: List[int] = []
    val_indices: List[int] = []

    for indices in label_to_indices.values():
        rng.shuffle(indices)
        val_count = int(len(indices) * val_fraction)
        if val_count == 0 and len(indices) > 0:
            val_count = 1
        val_indices.extend(indices[:val_count])
        train_indices.extend(indices[val_count:])

    train_dataset = cast(Dataset, dataset.select(train_indices))
    val_dataset = cast(Dataset, dataset.select(val_indices))
    return train_dataset, val_dataset


def _extract_logits(predictions) -> torch.Tensor:
    raw_logits = predictions.predictions
    if isinstance(raw_logits, (list, tuple)):
        raw_logits = raw_logits[0]
    if isinstance(raw_logits, np.ndarray):
        return torch.from_numpy(raw_logits)
    return torch.tensor(raw_logits)


def _collect_labels(dataset: Dataset, label_col: str, label2id: Dict[str, int]) -> np.ndarray:
    return np.array([label2id[str(label)] for label in dataset[label_col]], dtype=int)


def _compute_metrics(true_labels: np.ndarray, probabilities: np.ndarray, num_labels: int) -> Dict[str, float]:
    predicted_labels = probabilities.argmax(axis=-1)
    accuracy = float(accuracy_score(true_labels, predicted_labels))
    if probabilities.shape[1] == 2:
        f1 = float(f1_score(true_labels, predicted_labels, average="binary", pos_label=1))
        roc_auc = float(roc_auc_score(true_labels, probabilities[:, 1]))
    else:
        f1 = float(f1_score(true_labels, predicted_labels, average="macro"))
        roc_auc = float(roc_auc_score(true_labels, probabilities, multi_class="ovr", average="macro"))
    cross_entropy = float(log_loss(true_labels, probabilities, labels=list(range(num_labels))))

    return {
        "cross_entropy": cross_entropy,
        "accuracy": accuracy,
        "f1": f1,
        "roc_auc": roc_auc,
    }


def _get_backbone_layers(backbone: nn.Module) -> List[nn.Module]:
    if hasattr(backbone, "encoder") and hasattr(backbone.encoder, "layer"):
        return list(backbone.encoder.layer)
    if hasattr(backbone, "esm") and hasattr(backbone.esm, "encoder") and hasattr(backbone.esm.encoder, "layer"):
        return list(backbone.esm.encoder.layer)
    if hasattr(backbone, "model") and hasattr(backbone.model, "encoder") and hasattr(backbone.model.encoder, "layer"):
        return list(backbone.model.encoder.layer)
    if hasattr(backbone, "layers"):
        return list(backbone.layers)
    raise ValueError("Unsupported backbone: cannot locate transformer layers to freeze/unfreeze.")


def _freeze_backbone_layers(backbone: nn.Module, train_last_n: int) -> None:
    for param in backbone.parameters():
        param.requires_grad = False

    layers = _get_backbone_layers(backbone)
    if train_last_n <= 0:
        return

    if train_last_n > len(layers):
        train_last_n = len(layers)

    for layer in layers[-train_last_n:]:
        for param in layer.parameters():
            param.requires_grad = True


def main(args):
    os.makedirs(args.folder_params, exist_ok=True)

    print("Loading dataset...")
    train_dataset = cast(Dataset, load_dataset("csv", data_files=args.train_csv)["train"])

    if args.val_csv:
        val_dataset = cast(Dataset, load_dataset("csv", data_files=args.val_csv)["train"])
        label_source = concatenate_datasets([train_dataset, val_dataset])
    else:
        train_dataset, val_dataset = _stratified_split(
            train_dataset,
            args.column_labels,
            args.val_fraction,
            args.seed,
        )
        label_source = train_dataset

    label2id = _build_label_mapping(label_source, args.column_labels)

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, do_lower_case=False)

    def preprocess(example):
        tokenized = tokenizer(
            example[args.column_sequences],
            truncation=True,
            max_length=args.max_length,
        )
        tokenized["labels"] = label2id[str(example[args.column_labels])]
        return tokenized

    print("Tokenizing dataset...")
    tokenized_train = cast(Dataset, train_dataset.map(preprocess, remove_columns=train_dataset.column_names))
    tokenized_val = cast(Dataset, val_dataset.map(preprocess, remove_columns=val_dataset.column_names))

    model = ESMSequenceClassifier(
        backbone_name=args.backbone,
        num_labels=len(label2id),
        dropout=args.dropout,
    )
    _freeze_backbone_layers(model.backbone, args.train_last_n)

    for p in model.classifier.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable}/{n_total} ({100.0 * n_trainable / n_total:.2f}%)")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.folder_params,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        bf16=args.bf16,
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.patience),
        ],
    )

    print("Starting training...")
    trainer.train()

    print("Saving artifacts...")
    trainer.save_model(args.folder_params)
    model.backbone.save_pretrained(args.folder_params)
    tokenizer.save_pretrained(args.folder_params)
    torch.save(model.classifier.state_dict(), os.path.join(args.folder_params, "classifier_head.pt"))

    label_info = {
        "label2id": label2id,
        "id2label": {str(v): k for k, v in label2id.items()},
        "backbone": args.backbone,
        "train_last_n": args.train_last_n,
    }
    with open(os.path.join(args.folder_params, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=2)

    finetuned_test_metrics = None
    if args.test_csv:
        print("Evaluating on test set...")
        test_dataset = cast(Dataset, load_dataset("csv", data_files=args.test_csv)["train"])
        tokenized_test = cast(Dataset, test_dataset.map(preprocess, remove_columns=test_dataset.column_names))

        predictions = trainer.predict(tokenized_test)
        logits = _extract_logits(predictions)
        probabilities = F.softmax(logits, dim=-1).cpu().numpy()
        true_labels = predictions.label_ids
        finetuned_test_metrics = _compute_metrics(true_labels, probabilities, len(label2id))

        print("Test metrics:")
        print(json.dumps(finetuned_test_metrics, indent=2))

    print("Evaluating on validation set...")
    val_predictions = trainer.predict(tokenized_val)
    val_logits = _extract_logits(val_predictions)
    val_probabilities = F.softmax(val_logits, dim=-1).cpu().numpy()
    val_true_labels = val_predictions.label_ids
    finetuned_val_metrics = _compute_metrics(val_true_labels, val_probabilities, len(label2id))

    print("Validation metrics:")
    print(json.dumps(finetuned_val_metrics, indent=2))

    metrics_log = {
        "finetuned": {
            "validation": finetuned_val_metrics,
        },
    }
    if finetuned_test_metrics is not None:
        metrics_log["finetuned"]["test"] = finetuned_test_metrics

    metrics_path = os.path.join(args.folder_params, "metrics_log.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics_log, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
