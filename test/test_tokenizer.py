from transformers import AutoTokenizer


model_name_path = 'Qwen\\Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name_path}"
tokenizer = AutoTokenizer.from_pretrained(model_path,use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
def encode():
    messages = [
            {"role": "user", "content": "你叫什么名字？"},{"role": "assistant", "content": "我叫小呆"}
        ]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True  # Switches between thinking and non-thinking modes. Default is True.
    )
    # model_inputs = tokenizer([text], return_tensors="pt").to(base_model.device)
    print(text)
def decode():
    x = tokenizer.decode([151644, 8948, 198, 56568, 101909, 18830, 100364, 9370, 15235, 54599,
 102, 44934, 1773, 151645, 198, 151644, 872, 198, 32837, 50107,
 81705, 39426, 99738, 151645, 198, 151644, 77091, 198, 151667, 271,
  151668, 271, 104381, 3837, 100346, 104034, 105651, 101486, 1773, 151645,
  198, 151645, 151645, 151645, 151645, 151645, 151645])
    print(x)
decode()


