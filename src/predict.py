from peft import PeftModel
from transformers import AutoModelForCausalLM,AutoTokenizer

from src import config

model_name = 'Qwen/Qwen3-0.6B'
model_name_path = 'Qwen\\Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name_path}"
trained_dir = f"I:\\PycharmProjects\\model-test\\model_trained\\{model_name_path}\\checkpoint-15"
merged_model_dir = f"I:\\PycharmProjects\\model-test\\model_trained\\{model_name_path}\\FP16"
## 加载peft_model
def load_peft_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=config.BNB_CONFIG,
        device_map="cuda:0"

    )
    # PEFT 并不是“复制模型” 而是in-place的，如果要复制使用copy.deepcopy(base_model)
    peft_model = PeftModel.from_pretrained(base_model,trained_dir)
    peft_model.print_trainable_parameters()
    ## trainable params: 0 || all params: 752,205,824 || trainable%: 0.0000
    ## trainable params: 573,440 || all params: 752,205,824 || trainable%: 0.0762
    ## trainable params: 0 || all params: 752,062,464 || trainable%: 0.0000
    ## trainable params: 430,080 || all params: 752,062,464 || trainable%: 0.0572
    print(peft_model.active_adapter)
    return peft_model
def load_base_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        merged_model_dir,
        quantization_config=config.BNB_CONFIG,
        device_map="cuda:0"

    )
    return base_model
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
model= load_base_model()
print(model.device)

def generate(prompt):
    messages = [
        {"role": "system", "content": "你是一个有帮助的 AI 助手。"},
        {"role": "user", "content": prompt}
    ]
    # text = tokenizer.apply_chat_template(
    #     messages,
    #     tokenize=False,
    #     add_generation_prompt=True,
    #     enable_thinking=False  # Switches between thinking and non-thinking modes. Default is True.
    # )
    # print(f"text:{text}")
    # model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)
    model_inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        return_tensors="pt",
        enable_thinking=False,
        add_generation_prompt=True,
    )
    model_inputs = model_inputs.to(model.device)
    # conduct text completion
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=32768
    )
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    return output_ids

def parse(output_ids):
    # parsing thinking content
    try:
        # rindex finding 151668 (</think>)
        index = len(output_ids) - output_ids[::-1].index(151668)
    except ValueError:
        index = 0

    thinking_content = tokenizer.decode(output_ids[:index], skip_special_tokens=True).strip("\n")
    content = tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")

    print("thinking content:", thinking_content)
    print("content:", content)
while True :
    prompt = input("请输入：___")
    # parse(generate(prompt))
    print(tokenizer.decode(generate(prompt), skip_special_tokens=True).strip("\n"))