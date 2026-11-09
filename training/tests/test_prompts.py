import prompts

RECORD = {"instruction": "Summarize.", "input": "The abstract.", "output": "The tldr."}


class FakeTokenizer:
    """Maps each character to one id so lengths are easy to reason about."""

    eos_token = "$"

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is False
        text = "".join(f"[{m['role']}]{m['content']}" for m in messages)
        return text + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens):
        assert add_special_tokens is False
        return {"input_ids": [ord(c) for c in text]}


def test_prompt_holds_the_instruction_and_abstract():
    message = prompts.prompt_messages(RECORD)[0]
    assert message["role"] == "user"
    assert "Summarize." in message["content"]
    assert "The abstract." in message["content"]


def test_prompt_never_contains_the_answer():
    messages = prompts.prompt_messages(RECORD)
    assert all("The tldr." not in m["content"] for m in messages)
    assert all(m["role"] != "assistant" for m in messages)


def test_only_the_answer_is_trained_on():
    encoded = prompts.encode(RECORD, FakeTokenizer(), 2048)
    trained = [label for label in encoded["labels"] if label != -100]
    assert "".join(chr(i) for i in trained) == "The tldr.$"


def test_prompt_tokens_are_masked_from_the_loss():
    encoded = prompts.encode(RECORD, FakeTokenizer(), 2048)
    prompt_length = len(encoded["input_ids"]) - len("The tldr.$")
    assert encoded["labels"][:prompt_length] == [-100] * prompt_length


def test_input_ids_labels_and_mask_have_equal_length():
    encoded = prompts.encode(RECORD, FakeTokenizer(), 2048)
    assert len(encoded["input_ids"]) == len(encoded["labels"]) == len(encoded["attention_mask"])
    assert set(encoded["attention_mask"]) == {1}


def test_encoding_respects_max_length():
    encoded = prompts.encode(RECORD, FakeTokenizer(), 10)
    assert len(encoded["input_ids"]) == 10
    assert len(encoded["labels"]) == 10
