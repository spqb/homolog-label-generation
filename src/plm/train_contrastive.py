import argparse
import torch
import os
import sys
import time
from transformers import AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# Add parent directory to path to import train modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from train.dataloaders.dataloader import PairDataset, PairwiseInputCollator
from train.models.contrastive import ContrastiveLM
from train.trainer import Trainer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Encodes sequences using a pre-trained language model and optionally fine-tunes it before encoding the sequences.")
    parser.add_argument("--backbone", type=str, default="facebook/esm2_t33_650M_UR50D", help="Model name.", choices=["Rostlab/prot_bert", "facebook/esm2_t33_650M_UR50D"])
    parser.add_argument("--train_csv", type=str, default=None, help="Path to the training dataset in .csv format.")
    parser.add_argument("--folder_params", type=str, default=None, help="Output directory where to save the model's parameters.")
    parser.add_argument("--column_sequences", type=str, default="sequence", help="Column name in the input .csv file containing the sequences.")
    parser.add_argument("--column_labels", type=str, default="label", help="Column name in the input .csv file containing the labels.")
    parser.add_argument("--column_headers", type=str, default="header", help="Column name in the input .csv file containing the sequence identifiers.")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Maxium number of epochs.")
    parser.add_argument("--save_steps", type=int, default=50, help="Save the model every N steps.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--feat_dim", type=int, default=128, help="Feature dimension for the contrastive heads.")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--patience", type=int, default=5, help="Patience for early stopping.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of gradient accumulation steps.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the decomposition.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability.")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 precision.")
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging.")
    
    return parser


def main(config):
    
    if config["folder_params"] is not None:
        if not os.path.exists(config["folder_params"]):
            os.makedirs(config["folder_params"])
    
    if config["train_csv"] is not None:
        assert os.path.exists(config["train_csv"]), f"Training dataset {config['train']} does not exist."
        print("Loading training dataset...")
        train_dataset = PairDataset(config["train_csv"], column_sequences=config["column_sequences"], column_labels=config["column_labels"])
        print(f"Constructed {len(train_dataset)} positive pairs from the input dataset")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(config["backbone"], do_lower_case=False)
    
    # LoRA configuration
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,
        inference_mode=False,
        r=int(config["lora_rank"]),
        lora_alpha=int(config["lora_alpha"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        target_modules=["query", "value"],
    )
    
    print("Loading model...")
    model = ContrastiveLM(feat_dim=config["feat_dim"], backbone=config["backbone"])
    model = model.to(device)
    spaced_tokens = True if "prot_bert" in config["backbone"] else False
 
    model.backbone = get_peft_model(model.backbone, lora_config)
    model.backbone.print_trainable_parameters()
    model.train()
    collator_fn = PairwiseInputCollator(tokenizer, max_length=int(config["max_length"]), insert_whitespace=spaced_tokens)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"]))

    print("Starting training...")
    start_time_train = time.time()
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        output_dir=config["folder_params"],
        optimizer=optimizer,
        collator_fn=collator_fn,
        num_train_epochs=int(config["epochs"]),
        batch_size=int(config["batch_size"]),
        learning_rate=float(config["lr"]),
        patience=int(config["patience"]),
        save_steps=int(config["save_steps"]),
        gradient_accumulation_steps=int(config["gradient_accumulation_steps"]),
        bf16=bool(config["bf16"]),
        wandb=bool(config["wandb"]),
    )
    trainer.train()
    model = trainer.model
    time_train = time.time() - start_time_train
    print(f"Training completed in {time_train:.2f}s") 
          
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args)
    main(config)