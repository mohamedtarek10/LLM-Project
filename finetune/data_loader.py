import os
from typing import Any

from datasets import load_dataset

# Define a standard Alpaca-style prompt template
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Input:
{input}

### Response:
{output}"""

NO_INPUT_PROMPT_TEMPLATE = """Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
{instruction}

### Response:
{output}"""


def format_sample(sample: dict[str, Any]) -> str:
    """
    Format a data sample into a prompt for instruction tuning.
    Expects keys: 'instruction', 'input' (optional), 'output'.
    """
    instruction = sample.get("instruction", "")
    input_text = sample.get("input", "")
    output = sample.get("output", "")

    if input_text:
        text = ALPACA_PROMPT_TEMPLATE.format(
            instruction=instruction, input=input_text, output=output
        )
    else:
        text = NO_INPUT_PROMPT_TEMPLATE.format(instruction=instruction, output=output)

    return (
        text + "\n<|endoftext|>"
    )  # Add EOS token if needed by the specific tokenizer, or rely on trainer


def load_training_data(data_path: str, validation_split_percentage: float = 0.1):
    """
    Load dataset from a file (json/jsonl) and split into train/val.
    """
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found at: {data_path}")

    # Load dataset
    ext = data_path.split(".")[-1]
    if ext == "jsonl":
        dataset = load_dataset("json", data_files=data_path)
    elif ext == "json":
        dataset = load_dataset("json", data_files=data_path)
    else:
        # Fallback to text lines or use load_dataset defaults
        dataset = load_dataset("text", data_files=data_path)

    # If the dataset didn't have a split, it's usually in 'train'
    full_dataset = dataset["train"]

    # Apply formatting
    # We map over the dataset to create a 'text' column which SFTTrainer expects
    formatted_dataset = full_dataset.map(lambda x: {"text": format_sample(x)})

    # Split
    if validation_split_percentage > 0:
        dataset_split = formatted_dataset.train_test_split(test_size=validation_split_percentage)
        return dataset_split["train"], dataset_split["test"]

    return formatted_dataset, None
