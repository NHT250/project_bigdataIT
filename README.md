# TechPulse

A real-time tech news assistant built on papers from arXiv and stories from Hacker News, for a Big Data course at Van Lang University.

| Part | Status |
|---|---|
| Data collection (arXiv, Hacker News, PostgreSQL) | Done, runs continuously |
| Fine-tuned summarizer (Qwen 2.5 1.5B, QLoRA) | Done, beats the baseline |
| RAG question answering | Not built yet |
| Website | Not built yet |

## How it works

```
arXiv API ─────────┐
                   ├─> collector ─> PostgreSQL (articles)
HN Firebase API ───┘

SciTLDR ─> training/ ─> models/summarizer (kept on the local machine, not in git)
```

- The collector uses only the official APIs and parses no HTML. Every request sends an honest User-Agent with a contact address, arXiv gets at most one request every 3 seconds, and network errors back off exponentially.
- Articles are upserted on `(source, source_id)`, so repeated polling doesn't create duplicates. The arXiv `source_id` leaves out the version number.
- Each source polls in its own thread, and a failed poll doesn't stop the loop.

## Run the collector

Needs Docker with Compose.

```bash
cp .env.example .env
```

Set `POSTGRES_PASSWORD` and `CONTACT_EMAIL` in `.env`, then:

```bash
docker compose up -d --build
```

The stack is isolated from any other Compose project on the machine:

| Resource | Value |
|---|---|
| Project | `techpulse` |
| Network | `techpulse_default` |
| Volume | `techpulse_pgdata` |
| PostgreSQL | `127.0.0.1:5433` |
| API (reserved) | `8001` |

Memory is capped at 256 MB for PostgreSQL and 128 MB for the collector.

## Train the summarizer

Tested on Windows in Git Bash, with Python 3.12, [uv](https://docs.astral.sh/uv/) and an RTX 4050 6 GB. The run took 32 minutes and peaked at 2.04 GB of VRAM. Keep the virtual environment outside the repository if its path is long, because loading the torch DLLs can fail with WinError 206.

Set up the environment:

```bash
uv venv --python 3.12 "$HOME/.venvs/techpulse"
```

```bash
export VIRTUAL_ENV="$HOME/.venvs/techpulse"
```

```bash
uv pip install -r training/requirements.txt
```

```bash
uv pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu130 --reinstall-package torch
```

Download [SciTLDR-A](https://github.com/allenai/scitldr):

```bash
mkdir -p data/scitldr && for f in train dev test; do curl -fsSL -o "data/scitldr/$f.jsonl" "https://raw.githubusercontent.com/allenai/scitldr/master/SciTLDR-Data/SciTLDR-A/$f.jsonl"; done
```

Prepare the data, train, then score on the test split:

```bash
"$VIRTUAL_ENV/Scripts/python.exe" training/prepare.py
```

```bash
"$VIRTUAL_ENV/Scripts/python.exe" training/train.py --batch-size 2 --max-seq-length 2048 --epochs 3
```

```bash
"$VIRTUAL_ENV/Scripts/python.exe" training/evaluate.py
```

## Results

Mean ROUGE F1 on the 618 examples of the SciTLDR-A test split, taking the best reference for each paper. The baseline is the first two sentences of the abstract. The source file is [training/results/test_metrics.json](training/results/test_metrics.json).

| Metric | Model | Baseline | Gain |
|---|---|---|---|
| ROUGE-1 | 0.391 | 0.310 | +26% |
| ROUGE-2 | 0.163 | 0.117 | +40% |
| ROUGE-L | 0.322 | 0.228 | +41% |

The kept model is from epoch 1 of 3, because validation loss rose after it (1.930, 1.968, 2.142).

## Where the model is

The model is not in git. Training writes the LoRA adapter to `models/summarizer/` in the repository root (527 MB), and the 4-bit base model, `unsloth/Qwen2.5-1.5B-Instruct-unsloth-bnb-4bit`, comes from the Hugging Face cache. On another machine, retrain with the commands above.

Load it with plain `transformers` and `peft`, as `training/evaluate.py` does. Unsloth's fast inference path failed on the training machine with `CUBLAS_STATUS_EXECUTION_FAILED`.

## Limitations

1. The model was trained on SciTLDR, not on the articles collected here. This is transfer learning: it learns to summarize from a public dataset and is then applied to collected papers.
2. The train, dev and test sets are SciTLDR's own split, not a time-based split.
3. The model writes one-sentence summaries, not two or three sentences.
4. The model has not been compared with the un-tuned Qwen base model, so the gain from fine-tuning alone is unmeasured.

## Tests

Collector tests run against a real PostgreSQL in Docker:

```bash
CONTACT_EMAIL=test@example.invalid docker compose --profile test run --rm --build tests
```

Training tests:

```bash
PYTHONPATH=training "$HOME/.venvs/techpulse/Scripts/python.exe" -m pytest -q -p no:cacheprovider training/tests
```
