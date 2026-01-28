from transformers import AutoTokenizer

# 从 slow tokenizer 生成 fast tokenizer
model_name = 'Qwen/Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer.save_pretrained(model_path)  # 会生成 tokenizer.json
# tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
