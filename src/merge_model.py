import os

import torch
from peft import PeftModel
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM

from src import config

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
model_name = 'Qwen/Qwen3-0.6B'
model_name_path = 'Qwen\\Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name_path}"
#
# {
#     "epoch": 18.0,
#     "grad_norm": 3.2750186920166016,
#     "learning_rate": 0.00017333333333333334,
#     "loss": 0.9467509388923645,
#     "step": 18
# }
trained_dir = f"{str(config.MODEL_TRAINED)}\\{model_name_path}\\checkpoint-18"
FP16_DIR = f"{str(config.MODEL_TRAINED)}\\{model_name_path}\\FP16"
print(FP16_DIR)
## 加载FP16模型
tokenizer = AutoTokenizer.from_pretrained(model_path,trust_remote_code=True)
model_qw3 = AutoModelForCausalLM.from_pretrained(model_path,
                                                 device_map="cuda:0",
                                                 dtype=torch.float16)
model_qw3.eval()

## 加载Lora

model_qw3_lora = PeftModel.from_pretrained(model_qw3,
                                           trained_dir,
                                           dtype=torch.float16
                                           # ,is_trainable=True
                                           )
# 合并适配器和基础模型
merged_model = model_qw3_lora.merge_and_unload()
# 合并保存
tokenizer.save_pretrained(FP16_DIR)
merged_model.save_pretrained(FP16_DIR)