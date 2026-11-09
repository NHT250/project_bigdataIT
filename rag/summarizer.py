"""Run the fine-tuned model with plain transformers + peft.

Unsloth's fast inference path fails on this GPU setup.
"""
import sys

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

from rag.settings import ROOT

sys.path.insert(0, str(ROOT / "training"))
import prepare  # noqa: E402
import prompts  # noqa: E402

MODEL_DIR = ROOT / "models" / "summarizer"


class Summarizer:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
        self.model = AutoPeftModelForCausalLM.from_pretrained(str(MODEL_DIR), device_map="cuda")
        self.model.eval()

    def __call__(self, content):
        record = {"instruction": prepare.INSTRUCTION, "input": content}
        text = prompts.build_prompt(record, self.tokenizer)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=80, do_sample=False)
        return self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True
        ).strip()
