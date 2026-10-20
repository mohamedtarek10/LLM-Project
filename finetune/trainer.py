import argparse

import torch
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments
from trl import SFTTrainer

from finetune.data_loader import load_training_data


def train(
    model_name: str,
    data_path: str = "ev_charging_dataset.json",
    output_dir: str = "./results",
    batch_size: int = 4,
    grad_accum_steps: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 2e-4,
    use_wandb: bool = False,
):
    print(f"Starting training for {model_name}...")

    # 1. Load Data
    train_dataset, eval_dataset = load_training_data(data_path)
    print(f"Loaded {len(train_dataset)} training samples.")

    # 2. Config Quantization (QLoRA)
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=False,
    )

    # 3. Load Model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config, device_map="auto", trust_remote_code=True
    )

    model.config.use_cache = False  # Silence warnings
    model.config.pretraining_tp = 1

    # Prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # 4. Load Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16

    # 5. LoRA Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    # 6. Training Arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum_steps,
        optim="paged_adamw_32bit",
        save_steps=50,
        logging_steps=10,
        learning_rate=learning_rate,
        weight_decay=0.001,
        fp16=True,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
        report_to="wandb" if use_wandb else "none",
    )

    # 7. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        dataset_text_field="text",
        max_seq_length=512,  # Adjust based on GPU memory
        tokenizer=tokenizer,
        args=training_args,
        packing=False,
    )

    # 8. Train
    print("Training...")
    trainer.train()

    # 9. Save Model
    print(f"Saving model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune a model using QLoRA")
    parser.add_argument("--model_name", type=str, default="mistral:latest", help="Model identifier")
    parser.add_argument(
        "--data_path", type=str, required=True, help="Path to training data (jsonl)"
    )
    parser.add_argument("--output_dir", type=str, default="./results", help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Enable WandB logging")

    args = parser.parse_args()

    train(
        model_name=args.model_name,
        data_path=args.data_path,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
    )
