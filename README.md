# TechPulse

A real-time tech news assistant built on papers from arXiv and stories from Hacker News, for a Big Data course at Van Lang University.

| Part | Status |
|---|---|
| Data collection (arXiv, Hacker News, PostgreSQL) | Done, runs continuously |
| Fine-tuned summarizer (Qwen 2.5 1.5B, QLoRA) | Done, beats the baseline |
| RAG question answering | Done, tested against the live Gemini API |
| Website | Done, runs on port 8001 |

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

## Ask questions (RAG)

The RAG code runs on the host, because it needs the GPU. It adds to the environment from "Train the summarizer":

```bash
uv pip install -r rag/requirements.txt
```

Put `GEMINI_API_KEY` in `.env`, then index the articles and ask. Both jobs only process what's new, so run them again to pick up newly collected articles.

```bash
PYTHONPATH=. "$VIRTUAL_ENV/Scripts/python.exe" -m rag.index embed
```

```bash
PYTHONPATH=. "$VIRTUAL_ENV/Scripts/python.exe" -m rag.index summarize
```

```bash
PYTHONPATH=. "$VIRTUAL_ENV/Scripts/python.exe" -m rag.ask "What is new in graph neural networks?"
```

- Articles are split into 500-token chunks with a 50-token overlap and embedded with all-MiniLM-L6-v2. Search takes the 5 closest chunks by cosine similarity in pgvector.
- Gemini answers only from those chunks and cites them as `[n]`. An answer with no citations or a citation to a source that doesn't exist is rejected instead of shown.
- The default model is `gemini-3.6-flash`, because `gemini-2.0-flash` has been shut down. Set `GEMINI_MODEL` to use another one.
- A 429 from the API is retried up to 5 times, waiting as long as the API asks or backing off exponentially from 2 seconds, with no wait longer than 60 seconds.
- On Gemini's free tier, Google may use what is sent to improve its products. Here that is the question and the retrieved article text.
- Hacker News stories are indexed by title only, and only arXiv papers get a model summary, because the summarizer was trained on paper abstracts.
- The summarize job can stop with CUDA errors on the RTX 4050 laptop GPU, and the NVIDIA driver logged errors during those crashes. Finished summaries are saved, so running the job again continues from where it stopped.
- If the GPU fails while embedding, the embedder reloads on the CPU and carries on.

## Website

FastAPI serves three plain HTML and JavaScript pages, with no build step. It runs on the host, next to the RAG code:

```bash
uv pip install -r web/requirements.txt
```

```bash
"$VIRTUAL_ENV/Scripts/python.exe" -m uvicorn web.app:app --host 127.0.0.1 --port 8001
```

Then open http://127.0.0.1:8001.

- **Feed** shows the newest papers and stories with their summaries, can be filtered by source, and polls every 10 seconds. With both sources selected they take turns, because arXiv publishes in batches and would otherwise be pushed out by Hacker News.
- **Chat** answers from the archive with numbered links and shows how many articles it searched.
- **Search** finds articles by keyword with PostgreSQL full-text search.

The collector adds new articles every 5 minutes and they appear in Feed and Search straight away. Chat and the summaries only include them after the two `rag.index` jobs have run again.

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

RAG tests, which need the PostgreSQL container running:

```bash
PYTHONPATH=. "$HOME/.venvs/techpulse/Scripts/python.exe" -m pytest -q -p no:cacheprovider rag/tests
```

Website tests, which also need the PostgreSQL container running:

```bash
PYTHONPATH=. "$HOME/.venvs/techpulse/Scripts/python.exe" -m pytest -q -p no:cacheprovider web/tests
```
