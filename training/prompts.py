"""Prompt format shared by training and inference."""


def prompt_messages(record):
    return [{"role": "user", "content": f"{record['instruction']}\n\n{record['input']}"}]


def build_prompt(record, tokenizer):
    return tokenizer.apply_chat_template(
        prompt_messages(record), tokenize=False, add_generation_prompt=True
    )


def encode(record, tokenizer, max_length):
    """Tokenise one example; only the answer contributes to the loss."""
    prompt_ids = tokenizer(build_prompt(record, tokenizer), add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(record["output"] + tokenizer.eos_token, add_special_tokens=False)["input_ids"]
    input_ids = (prompt_ids + answer_ids)[:max_length]
    labels = ([-100] * len(prompt_ids) + answer_ids)[:max_length]
    return {"input_ids": input_ids, "labels": labels, "attention_mask": [1] * len(input_ids)}
