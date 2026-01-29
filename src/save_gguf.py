import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from src import config


model_name_path = 'Qwen\\Qwen3-0.6B'
merged_model_dir = f"{config.MODEL_TRAINED}\\{model_name_path}\\FP16"


from unsloth import FastLanguageModel
import torch

# 加载模型和tokenizer
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = merged_model_dir,  # 替换为你的模型路径或Hugging Face模型ID
    load_in_4bit = True,             # 使用4bit量化加载
)
### unsloth和torch2.5.1不兼容，需要torch2.10.0以上版本，不能升级，会不支持cuda
# 保存为GGUF格式
model.save_pretrained_gguf(
    str(config.MODEL_GGUF/"qwen3-ft-fp16-q4km"),                  # 输出目录
    tokenizer,                       # tokenizer对象
    quantization_method = "q4_k_m"   # 量化方法，可选：q4_k_m, Q8_0, f16等
)

print("模型已成功保存为GGUF格式")