# TechPulse

A real-time tech news assistant built on papers from arXiv, stories from Hacker News and practical articles from Dev.to, for a Big Data course at Van Lang University.

| Part | Status |
|---|---|
| Data collection (arXiv, Hacker News, Dev.to, PostgreSQL) | Done, runs continuously |
| Fine-tuned summarizer (Qwen 2.5 1.5B, QLoRA) | Done, beats the baseline |
| RAG question answering | Done, tested against the live Gemini API |
| Website | Done, runs on port 8001 |

## How it works

```
arXiv API ─────────┐
                   ├─> collector ─> PostgreSQL (articles)
HN Firebase API ───┤
Linked pages ──────┤  (Hacker News story bodies, see below)
Dev.to API ────────┘

SciTLDR ─> training/ ─> models/summarizer (kept on the local machine, not in git)
```

Three sources, three kinds of content, and three ways of getting the text:

| Source | Kind | How the text is obtained |
|---|---|---|
| arXiv | Academic research | Official API, which includes the abstract |
| Hacker News | News and discussion | Official API for metadata, trafilatura for the linked page |
| Dev.to | Hands-on technical articles | Official API, which returns the article as Markdown |

- arXiv papers, Hacker News stories and Dev.to articles come from the official APIs. Every request sends an honest User-Agent with a contact address, arXiv gets at most one request every 3 seconds, and network errors back off exponentially.
- Articles are upserted on `(source, source_id)`, so repeated polling doesn't create duplicates. The arXiv `source_id` leaves out the version number.
- Each source polls in its own thread, and a failed poll doesn't stop the loop. Now the archive is complete, polls are light: Dev.to every 15 minutes, Hacker News every 20 and arXiv every 60, so the Feed still shows new articles arriving.
- A 429 from a source halves that source's request rate until the collector restarts, at most once a minute (requests already on their way come back 429 together) and never below one request in 10 seconds. `Retry-After` is honoured, as seconds or as a date. The rate is never raised again, because probing for the limit would mean drawing more 429s.

### Dev.to articles

Dev.to's public API needs no key. The collector reads the tags `ai`, `machinelearning`, `python`, `webdev`, `programming` and `datascience` every 15 minutes from `GET /api/articles?tag=…&per_page=100&page=N`, newest first, and stops a tag at the first page with nothing new or nothing from the last 7 days. The listing has no body, so each new article costs one more request to `GET /api/articles/{id}` for its `body_markdown`, which is stored as the article's content as it is.

- All Dev.to requests share one limiter: at most 5 a second, retries included. Bodies are read one at a time, about 1.4 requests a second, because Dev.to publishes no limit and eight at once drew 429s within minutes (11/2026), while one at a time ran for hours without one. Network errors back off exponentially; an article that still fails is tried again on the next poll, and the poll goes on with the rest.
- An article that appears under several tags is fetched and stored once. Re-polling updates reaction and comment counts and keeps the stored body.
- An article without `body_markdown` is skipped and not fetched again while the collector runs.
- Dev.to answers 403 to the full User-Agent (`Van Lang University course project`), so its requests say `VLU course project` instead, with the same contact address.
- The first poll takes the last 7 days. To take it without restarting the collector, run one poll by hand: `docker compose run --rm --no-deps collector python -m techpulse.devto`.
- Dev.to articles get no model summary. The summarizer was fine-tuned on scientific paper abstracts (SciTLDR), and practical articles in Markdown, with code blocks and steps, are a different kind of text, so its summaries would be worse. The Feed shows the author's own description from the API instead, labelled "Author's description".

### Hacker News story bodies

The Hacker News API gives a story's title and link but not the article, so a third thread gives each story a body once it is an hour old, 40 stories a minute. The body has up to three labelled parts: the post's own text (Ask HN and similar, from the API), the main text of the linked page, and the first five live comments (from the API). Markup in API text is turned into plain text with Python's standard library.

Reading the linked page is the one place the collector fetches a page that isn't an API, so it follows these rules:

- It reads the site's `robots.txt` first and obeys it, including rules for `TechPulse` by name and `Crawl-delay`. A site whose `robots.txt` can't be fetched is left alone.
- It waits at least 5 seconds between requests to the same site, and follows redirects one hop at a time so every site's `robots.txt` is checked.
- A refused, paywalled, non-HTML or larger than 2 MB page is skipped, never retried or worked around. Each story's outcome is kept in the `hn_content` table, so no page is read twice.
- [trafilatura](https://trafilatura.readthedocs.io/) keeps only the main text, capped at 20,000 characters. The text is used only for search and answers; the website shows titles, summaries and links back to the source.

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

Memory is capped at 256 MB each for PostgreSQL and the collector.

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

Put `GROQ_API_KEY` and/or `GEMINI_API_KEY` in `.env`, then index the articles and ask. Both jobs only process what's new, so run them again to pick up newly collected articles.

```bash
PYTHONPATH=. "$VIRTUAL_ENV/Scripts/python.exe" -m rag.index embed
```

```bash
PYTHONPATH=. "$VIRTUAL_ENV/Scripts/python.exe" -m rag.index summarize
```

```bash
PYTHONPATH=. "$VIRTUAL_ENV/Scripts/python.exe" -m rag.ask "What is new in graph neural networks?"
```

- Articles are split into 500-token chunks with a 50-token overlap and embedded with all-MiniLM-L6-v2.
- Retrieval is hybrid by default. One branch ranks articles by their closest chunk in pgvector (cosine), the other by PostgreSQL full-text search (`ts_rank_cd` over a generated `search_document` column on `articles`, title weighted above body, with a GIN index). Each branch takes its top 20 articles and Reciprocal Rank Fusion (k = 60) keeps the best 5; it uses ranks only, because the two scores are on different scales. Each article is shown to the LLM by its chunk closest to the question. Before indexing and searching, a letter joined to a number is split off, so a question about "Qwen 2.5" finds an article that writes "QWen2.5", and the other way round. `RETRIEVAL_MODE=dense` or `bm25` uses one branch alone.
- `PYTHONPATH=. python -m rag.eval_retrieval` measures recall@5 and recall@10 of the three modes on the 20 questions in `rag/eval/retrieval_questions.json`.
- Groq answers first by default, with Gemini as the fallback. `LLM_PROVIDER=gemini` puts Gemini first; with only one key set, only that one is used. The Groq model is `qwen/qwen3.8-27b` (listed as active by `GET /openai/v1/models`, 11/2026). On the same questions it cited its sources every time and kept to the question's language, while `openai/gpt-oss-20b`, about as fast, often left out citations and once answered an English question in Thai. `GROQ_MODEL` picks another. Restart the website after changing these.
- A 429 or 503 from Groq is retried 3 times, waiting as long as Groq asks or 1, 2 then 4 seconds, never more than 5. If Groq still fails, or fails in any other way, the same question goes to Gemini without an error on the page; if Groq broke off mid-answer, the page drops the partial answer and Gemini writes it again. Only when both fail does the page show a short "please try again" note with a Try again button.
- Under each answer, Chat suggests up to 4 follow-up questions. One more LLM request writes 6 candidates from the passages the answer cited (an uncited passage can be off topic), cut to their first 700 characters, each tagged with the passage that answers it; a candidate is kept only if searching the archive for it finds that passage's article again, so a click doesn't end in "I don't have information". Suggestions are cached by question.
- Groq's free tier allows each model 1,000 requests a day and only 7,000 to 8,000 input tokens a minute. A question sends about 3,000 tokens of retrieved text, so a burst of more than two or three questions a minute reaches the limit and those answers come from Gemini.
- The LLM answers only from those chunks and cites them as `[n]`. An answer with no citations or a citation to a source that doesn't exist is rejected instead of shown.
- On the website, follow-up questions can refer to the last three answers. Gemini first rewrites a follow-up such as "Explain the first one" into a question that stands on its own, then the search and the answer use that question. This costs a second Gemini request per follow-up. The command line answers each question on its own.
- The default model is `gemini-3.6-flash`, because `gemini-2.0-flash` has been shut down. Set `GEMINI_MODEL` to use another one.
- A 429 from the API is retried up to 5 times, waiting as long as the API asks or backing off exponentially from 2 seconds, with no wait longer than 60 seconds. When the 429 says the model's daily quota is used up, it is reported at once instead, because the quota only resets the next day.
- The free tier allows `gemini-3.6-flash` 20 requests a day, and each model has its own quota. When it runs out, set `GEMINI_MODEL=gemini-3.5-flash-lite` in `.env` and restart the website.
- On Gemini's free tier, Google may use what is sent to improve its products. Here that is the question and the retrieved article text.
- Hacker News stories are indexed with their body once the collector has found one, and by title until then. Dev.to articles are indexed with their full Markdown. Only arXiv papers get a model summary, because the summarizer was trained on paper abstracts.
- When a story gets a body, `rag.index embed` indexes it again, because it re-embeds any article whose text changed.
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

- **Feed** shows the newest papers and stories with their summaries, can be filtered by source, and polls every 10 seconds. With all sources selected they take turns, because arXiv publishes in batches and would otherwise be pushed out by Hacker News and Dev.to.
- **Chat** answers from the archive with numbered links and shows how many articles it searched. When a follow-up is rewritten, it also shows the question it searched for. **Clear** deletes the conversation.
- **Search** finds articles by keyword with PostgreSQL full-text search.

Each page is a separate HTML file, so the browser's localStorage keeps what they show when moving between them: the last 20 chat turns, the last search and its results, and the Feed's source filter. Nothing is kept on the server. If the browser blocks localStorage, as some do in private browsing, the pages still work but start empty.

The collector adds new articles every 15 to 60 minutes, depending on the source, and they appear in Feed and Search straight away. Chat and the summaries only include them after the two `rag.index` jobs have run again.

## Chạy trên máy khác (Run on another machine)

Needs Docker with Compose. The images come from Docker Hub, so nothing is built locally. In Git Bash or any Linux or macOS shell, make an empty folder and run these from inside it.

**1. Download the Compose file and the settings template.**

```bash
curl -fLO https://raw.githubusercontent.com/NHT250/project_bigdataIT/main/docker-compose.prod.yml
```

```bash
curl -fL -o .env https://raw.githubusercontent.com/NHT250/project_bigdataIT/main/.env.example
```

**2. Fill in `.env`.** Set `GEMINI_API_KEY` (free from [Google AI Studio](https://aistudio.google.com/apikey)), `CONTACT_EMAIL`, and `POSTGRES_PASSWORD`. Use only letters and digits in the password, because it goes into a connection URL.

**3. Download the model.** It is not in the Docker images. Feed, Search and Chat work without it, so skip this step on a machine with no NVIDIA GPU.

```bash
mkdir -p models/summarizer
```

```bash
curl -fL -o model.zip https://github.com/NHT250/project_bigdataIT/raw/model-files/techpulse-summarizer.zip
```

```bash
unzip -o model.zip -d models/summarizer
```

**4. Start it.** The first run downloads about 7 GB of images when unpacked.

```bash
docker compose -f docker-compose.prod.yml up -d
```

**5. Restore the collected articles (optional).** Without this, the database starts empty and fills up as new articles arrive.

```bash
curl -fL -o techpulse.dump https://github.com/NHT250/project_bigdataIT/raw/model-files/techpulse.dump
```

```bash
docker compose -f docker-compose.prod.yml stop ingest api indexer
```

```bash
docker compose -f docker-compose.prod.yml exec -T postgres pg_restore -U techpulse -d techpulse --clean --if-exists --no-owner --exit-on-error < techpulse.dump
```

```bash
docker compose -f docker-compose.prod.yml up -d
```

**6. Open http://localhost:8001.** The site answers about 30 seconds after the containers start.

### Notes

- The `1.0` images on Docker Hub predate Hacker News story bodies, Dev.to and follow-up questions in Chat. They need rebuilding and pushing again to include them.
- The `indexer` service embeds new articles for Chat every 5 minutes, then writes summaries. Without an NVIDIA GPU or the model it logs `Skipping summaries` and carries on, so new papers appear without a summary.
- To write summaries on a machine with an NVIDIA GPU and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), start with both files. The first summary also downloads the 4-bit Qwen base model, about 1 GB, from Hugging Face. Inside Docker this has only been checked as far as the container seeing the GPU.

```bash
docker compose -f docker-compose.prod.yml -f docker-compose.prod.gpu.yml up -d
```

- From a clone of this repository, `make dump` and `make restore` save and load `backup/techpulse.dump`. Without `make`, run `sh scripts/dump.sh` and `sh scripts/restore.sh`.
- Run only one copy of the collector per network, so arXiv still gets at most one request every 3 seconds.

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
