from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = 'Qwen/Qwen3-0.6B'
model_path = f"I:\\PycharmProjects\\model-test\\model_huggingface\\{model_name}"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="half",
    device_map="cuda:0"
)
model.save_pretrained(model_path)