"""QLoRA fine-tune of Qwen 2.5 1.5B on SciTLDR, sized for a 6 GB GPU.

If the GPU runs out of memory, retry with --batch-size 1 first and only then
lower --max-seq-length.
"""
import argparse
import json
from pathlib import Path

import gpu_env  # noqa: F401  (must stay before unsloth)
from unsloth import FastLanguageModel

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, Trainer, TrainerCallback, TrainingArguments

import prompts

MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "prepared"
OUTPUT_DIR = ROOT / "models" / "summarizer"


class VramLogger(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        peak = torch.cuda.max_memory_reserved() / 1024**3
        print(f"step {state.global_step}: peak VRAM reserved {peak:.2f} GB", flush=True)


def load_split(split, tokenizer, max_length, limit):
    with open(DATA_DIR / f"{split}.jsonl", encoding="utf-8") as lines:
        records = [json.loads(line) for line in lines]
    return Dataset.from_list([prompts.encode(r, tokenizer, max_length) for r in records[:limit]])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None, help="use only the first N examples")
    parser.add_argument("--output", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME, max_seq_length=args.max_seq_length, load_in_4bit=True, dtype=None
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    trainer = Trainer(
        model=model,
        train_dataset=load_split("train", tokenizer, args.max_seq_length, args.limit),
        eval_dataset=load_split("dev", tokenizer, args.max_seq_length, args.limit),
        data_collator=DataCollatorForSeq2Seq(tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[VramLogger()],
        args=TrainingArguments(
            output_dir=str(args.output / "checkpoints"),
            per_device_train_batch_size=args.batch_size,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=4,
            num_train_epochs=args.epochs,
            learning_rate=2e-4,
            logging_steps=25,
            eval_strategy="epoch",
            save_strategy="epoch",
            save_total_limit=2,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_num_workers=0,
            report_to="none",
            seed=3407,
        ),
    )
    trainer.train()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"saved to {args.output}; peak VRAM {torch.cuda.max_memory_reserved() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
