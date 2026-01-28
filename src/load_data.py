def tokenize_function(example,tokenizer):
    # example["messages"] 是一个 list[dict]
    text = tokenizer.apply_chat_template(
        example["messages"],
        tokenize=False,
        add_generation_prompt=False
    )

    tokens = tokenizer(
        text,
        truncation=True,
        max_length=2048,
        padding="max_length"
    )

    tokens["labels"] = tokens["input_ids"].copy()
    return tokens

