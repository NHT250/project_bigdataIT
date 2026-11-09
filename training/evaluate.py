"""Score the fine-tuned model against the lead-2-sentences baseline on a held-out split."""
import argparse
import json
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer

import prompts
import scoring

ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = Path(__file__).resolve().parent / "results"


def summarize(model, tokenizer, record):
    text = prompts.build_prompt(record, tokenizer)
    inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False).to("cuda")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=80, do_sample=False)
    return tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "summarizer")
    parser.add_argument("--split", default="test")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    with open(ROOT / "data" / "prepared" / f"{args.split}.jsonl", encoding="utf-8") as lines:
        records = [json.loads(line) for line in lines][: args.limit]

    # Plain transformers + peft: Unsloth's fast inference path fails on this GPU setup.
    tokenizer = AutoTokenizer.from_pretrained(str(args.model))
    model = AutoPeftModelForCausalLM.from_pretrained(str(args.model), device_map="cuda")
    model.eval()

    predictions = []
    model_rows, baseline_rows = [], []
    for index, record in enumerate(records, 1):
        prediction = summarize(model, tokenizer, record)
        predictions.append({"prediction": prediction, "references": record["references"]})
        model_rows.append(scoring.score(prediction, record["references"]))
        baseline_rows.append(
            scoring.score(scoring.lead_baseline(record["sentences"]), record["references"])
        )
        if index % 50 == 0:
            print(f"{index}/{len(records)}", flush=True)

    model_mean = scoring.mean(model_rows)
    baseline_mean = scoring.mean(baseline_rows)
    report = {
        "split": args.split,
        "examples": len(records),
        "model": model_mean,
        "baseline": baseline_mean,
        "model_beats_baseline": scoring.beats(model_mean, baseline_mean),
    }
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{args.split}_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with open(RESULTS_DIR / f"{args.split}_predictions.jsonl", "w", encoding="utf-8") as out:
        for row in predictions:
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
