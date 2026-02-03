import os

from datasets import load_dataset
from peft import prepare_model_for_kbit_training, PromptTuningConfig, PromptEncoderConfig, TaskType, PromptTuningInit, \
    PeftModelForCausalLM, get_peft_model, PrefixTuningConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, \
    DataCollatorForLanguageModeling

from src import config



os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
model_name = 'Qwen/Qwen3-0.6B'
model_name_path = 'Qwen\\Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name_path}"
trained_dir = f"{str(config.MODEL_TRAINED)}\\{model_name_path}\\prompt-tuning"
tokenizer = AutoTokenizer.from_pretrained(model_path,device_map="cuda:0",)
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained(model_path,quantization_config=config.BNB_CONFIG,device_map="cuda:0",)

prepare_model_for_kbit_training(model)
## 拼接在embedding层输出,维度是[num_virtual_tokens,embedding_size]在batch维度广播
prompt_config = PromptTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    prompt_tuning_init_text="你是一个小学生",
    num_virtual_tokens=len(tokenizer("你是一个小学生")["input_ids"]),
    tokenizer_name_or_path=model_path
)
## 虚拟token经过训练的embedding层，再经过每层专有的训练过的Encoder，拼接到attention的词维度
prompt_encoder_config = PromptEncoderConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10
)
#精确定义版结论
# 在标准 Prefix Tuning 中：
# 不论输入 token 是什么，
# 对于同一个模型 checkpoint，
# 每一层 attention 在 K / V 前面拼接的前缀矩阵内容都是完全相同的。
#
# 这里的“相同”指的是：
#
# 数值完全一致
#
# 不依赖输入 token
#
# 不随 batch / prompt / position 改变
prefix_tuning_config = PrefixTuningConfig(
    task_type=TaskType.CAUSAL_LM,
    num_virtual_tokens=10,
    prefix_projection=False
)
peft_model = get_peft_model(model,peft_config=prompt_config)
print(peft_model)
peft_model.print_trainable_parameters()
## trainable params: 7,168 || all params: 751,639,552 || trainable%: 0.0010
train_args = TrainingArguments(
    output_dir=trained_dir,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=1,
    learning_rate=4e-4,
    num_train_epochs=30,
    logging_steps=3,
    save_strategy="steps",
    save_steps=5,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)
dataset = load_dataset(
    path ="json",
    data_files=str(config.DATA_PATH/"train3.json")
)
def tokenizer_fun(input_data):
    text_chat_template = tokenizer.apply_chat_template(
        input_data["messages"],
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False
    )
    tokens = tokenizer(text_chat_template, truncation=True,padding="max_length",max_length=2048,)
    tokens["labels"] = tokens["input_ids"].copy()
    return tokens
tokenized_dataset_dict = dataset.map(tokenizer_fun,remove_columns=dataset["train"].column_names)
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,mlm=False)
trainer = Trainer(
    model = peft_model,
    args = train_args,
    train_dataset=tokenized_dataset_dict["train"],
    data_collator = data_collator
)
trainer.train()

## 保存权重
model.save_pretrained(trained_dir)
tokenizer.save_pretrained(trained_dir)