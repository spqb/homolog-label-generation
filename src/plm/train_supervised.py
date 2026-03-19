import argparse
import json
import os
from typing import Dict, List, cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModel, AutoTokenizer
from transformers import TrainerCallback
from transformers import DataCollatorWithPadding, Trainer, TrainingArguments


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
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
    ):
        super().__init__()

        backbone = AutoModel.from_pretrained(backbone_name)
        hidden_size = int(backbone.config.hidden_size)
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION",
            target_modules=["query", "key", "value"],
        )
        self.backbone = get_peft_model(backbone, lora_config)
        self.num_labels = num_labels

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_labels),
        )
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


class SavePeftAdapterCallback(TrainerCallback):
    """Saves LoRA adapter weights in each Trainer checkpoint directory."""

    def on_save(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or not hasattr(model, "backbone"):
            return control
        if not hasattr(model.backbone, "save_pretrained"):
            return control

        ckpt_dir = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        model.backbone.save_pretrained(ckpt_dir)
        return control


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Supervised finetuning of ESM2 with LoRA adapters and mean-pooled token embeddings.")
    parser.add_argument("--backbone", type=str, default="facebook/esm2_t33_650M_UR50D", help="Backbone model identifier.")
    parser.add_argument("--train_csv", type=str, required=True, help="Path to training CSV.")
    parser.add_argument("--folder_params", type=str, required=True, help="Output directory.")

    parser.add_argument("--column_sequences", type=str, default="sequence", help="CSV sequence column.")
    parser.add_argument("--column_labels", type=str, default="label", help="CSV label column.")

    parser.add_argument("--batch_size", type=int, default=16, help="Per-device batch size.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum tokenized sequence length.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout used in classification head.")
    parser.add_argument("--lora_rank", type=int, default=8, help="LoRA rank (r).")
    parser.add_argument("--lora_alpha", type=int, default=16, help="LoRA alpha scaling.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="LoRA dropout.")
    parser.add_argument("--bf16", action="store_true", help="Enable bf16 training.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")

    return parser


def _build_label_mapping(dataset: Dataset, label_col: str) -> Dict[str, int]:
    labels = dataset[label_col]
    unique_labels: List[str] = sorted({str(x) for x in labels})
    return {label: i for i, label in enumerate(unique_labels)}


def main(args):
    os.makedirs(args.folder_params, exist_ok=True)

    print("Loading dataset...")
    dataset = cast(Dataset, load_dataset("csv", data_files=args.train_csv)["train"])

    label2id = _build_label_mapping(dataset, args.column_labels)
    id2label = {v: k for k, v in label2id.items()}

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
    tokenized_dataset = cast(Dataset, dataset.map(preprocess, remove_columns=dataset.column_names))

    model = ESMSequenceClassifier(
        backbone_name=args.backbone,
        num_labels=len(label2id),
        dropout=args.dropout,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Ensure classification head is trainable
    for p in model.classifier.parameters():
        p.requires_grad = True

    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {n_trainable}/{n_total} ({100.0 * n_trainable / n_total:.2f}%)")
    model.backbone.print_trainable_parameters()

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=args.folder_params,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        logging_steps=50,
        save_steps=500,
        save_strategy="steps",
        eval_strategy="no",
        bf16=args.bf16,
        seed=args.seed,
        remove_unused_columns=False,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
        callbacks=[SavePeftAdapterCallback()],
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
        "id2label": {str(k): v for k, v in id2label.items()},
        "backbone": args.backbone,
        "lora_rank": args.lora_rank,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
    }
    with open(os.path.join(args.folder_params, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_info, f, indent=2)

    print("Done.")


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)
