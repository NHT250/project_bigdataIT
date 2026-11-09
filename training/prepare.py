"""Convert SciTLDR-A into instruction-format JSONL.

The labels are human-written TLDRs. They are deliberately not sentences copied
from the abstract, so a lead-sentences baseline cannot win by matching itself.
"""
import json
from pathlib import Path

INSTRUCTION = "Summarize this paper in one sentence."
ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = ROOT / "data" / "scitldr"
OUTPUT_DIR = ROOT / "data" / "prepared"


def to_record(row):
    if not row["source"] or not row["target"]:
        return None
    return {
        "instruction": INSTRUCTION,
        "input": " ".join(row["source"]),
        "output": row["target"][0],
        "references": row["target"],
        "sentences": row["source"],
    }


def convert(source, destination):
    kept = 0
    with open(source, encoding="utf-8") as lines, open(destination, "w", encoding="utf-8") as out:
        for line in lines:
            record = to_record(json.loads(line))
            if record is not None:
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                kept += 1
    return kept


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for split in ("train", "dev", "test"):
        print(split, convert(SOURCE_DIR / f"{split}.jsonl", OUTPUT_DIR / f"{split}.jsonl"))
