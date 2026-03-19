import os
import argparse
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from datasets import load_dataset

def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Finetune a model on a dataset")
    parser.add_argument("--backbone", type=str, default="facebook/esm2_t33_650M_UR50D", help="Model name.", choices=["Rostlab/prot_bert", "facebook/esm2_t33_650M_UR50D"])
    parser.add_argument("--train_csv", type=str, required=True, help="Path to the training dataset in .csv format.")
    parser.add_argument("--folder_params", type=str, required=True, help="Output directory.")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs.")
    parser.add_argument("--max_length", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate.")
    parser.add_argument("--weight_decay", type=float, default=0.001, help="Weight decay.")
    parser.add_argument("--lora_rank", type=int, default=8, help="Rank of the decomposition.")
    parser.add_argument("--lora_alpha", type=int, default=16, help="Scaling factor.")
    parser.add_argument("--lora_dropout", type=float, default=0.05, help="Dropout probability.")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 precision.")
    return parser

def main(args):
    
    if not os.path.exists(args.folder_params):
        os.makedirs(args.folder_params)
    
    # Import and preprocess the dataset
    print("Loading dataset...")
    dataset = load_dataset("csv", data_files=args.train_csv)
    
    # protBERT requires the amino acids to be separated by spaces
    tokenizer = AutoTokenizer.from_pretrained(args.backbone, do_lower_case=False)
    
    if "prot_bert" in args.backbone:
        def add_spaces(x):
            return {"sequence" : " ".join(list(x["sequence"]))}
        dataset = dataset.map(add_spaces)
        
    # Tokenize the sequences
    def tokenize_function(examples):
        return tokenizer(examples["sequence"], padding="max_length", truncation=True, max_length=args.max_length)
    
    print("Tokenizing sequences...")
    dataset_tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["sequence", "header", "label"])
    
    # Load the model and set the LoRA configuration
    print("Loading model...")
    model = AutoModelForMaskedLM.from_pretrained(args.backbone)

    lora_config = LoraConfig(
        r=args.lora_rank,  # Rank of the decomposition
        lora_alpha=args.lora_alpha,  # Scaling factor
        lora_dropout=args.lora_dropout,  # Dropout probability
        bias="none",
        target_modules=["query", "value"]  # Apply LoRA to attention layers
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # prepare the trainer
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)
    training_args = TrainingArguments(
        output_dir=args.folder_params,
        overwrite_output_dir=True,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        logging_steps=100,
        weight_decay=args.weight_decay,
        learning_rate=args.lr,
        save_steps=100,
        eval_strategy="no",
        bf16=args.bf16,
        logging_dir=args.folder_params,
    )
            
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset_tokenized["train"],
    )
    
    trainer.train()
    
if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    main(args)