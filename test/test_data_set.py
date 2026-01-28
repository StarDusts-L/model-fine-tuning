from datasets import load_dataset
from transformers import AutoTokenizer

from src import config

model_name_path = 'Qwen\\Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name_path}"
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
dataset = load_dataset(
    "json",
    data_files= str(config.DATA_PATH/"train1.json")
)
def tokenize_function(example):
    # example["messages"] 是一个 list[dict]
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )
    print(f"格式化text:{text}")
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens


tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset["train"].column_names,
)
print(tokenized_dataset)