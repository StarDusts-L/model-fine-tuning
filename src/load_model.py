import os

import pandas as pd
import torch
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          BitsAndBytesConfig, TrainingArguments, Trainer, DataCollatorForLanguageModeling)
from peft import LoraConfig,get_peft_model,prepare_model_for_kbit_training
from datasets import load_dataset

from src import config, load_data
# CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
model_name = 'Qwen/Qwen3-0.6B'
model_name_path = 'Qwen\\Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name_path}"


## use_fast的作用
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path = model_path,
                                             quantization_config=config.BNB_CONFIG,
                                             device_map="cuda:0",)
# 冻结所有base参数
# LayerNorm转fp32
# 保证梯度能稳定回传
prepare_model_for_kbit_training(model)
## 打印下模型有哪些参数
for param_name,params in model.named_parameters():
    print(param_name)
# 设置LoRA
# 不改k_proj,改它会整体扰乱注意力空间
# 总结成一张脑内表
# 模块	作用	为什么调
# q_proj	注意力视角	高影响、稳定
# v_proj	信息内容	直接改语义
# up_proj	特征生成	改“想法”
# down_proj	特征筛选	控制输出
lora_config = LoraConfig(
    r = 4,
    lora_alpha = 10,
    lora_dropout=0.07,
    target_modules = ["q_proj", "v_proj"],
    bias="none",
    task_type="CAUSAL_LM",

)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()#trainable params: 573,440 || all params: 752,205,824 || trainable%: 0.0762

trained_dir = f"I:\\PycharmProjects\\model-test\\model_trained\\{model_name_path}"

training_args = TrainingArguments(
    output_dir=trained_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=4e-4,
    num_train_epochs=30,
    logging_steps=1,
    save_strategy="steps",
    save_steps=3,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none"
)
from datasets import load_dataset

dataset = load_dataset(
    "json",
    data_files= str(config.DATA_PATH/"train2.json")
)
def tokenize_function(example):
    # example["messages"] 是一个 list[dict]
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    print(f"格式化text:{text}")
    tokens = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

    tokens["labels"] = tokens["input_ids"].copy()
    decoded = tokenizer.decode(
        [id for id in tokens["labels"] if id != -100],
        skip_special_tokens=False
    )
    print(f"格式化decoded:{decoded}")
    return tokens


tokenized_dataset = dataset.map(
    tokenize_function,
    remove_columns=dataset["train"].column_names,
)
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # causal LM
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset["train"],
    data_collator=data_collator,
)
## 如果需要强调及时EOS，可以在计算LOSS时，把EOS的权重调大
trainer.train()

## 保存权重
model.save_pretrained(trained_dir)
tokenizer.save_pretrained(trained_dir)