"""Embed new articles and summarize new papers. Safe to run again at any time."""
import argparse
import logging

import psycopg

from rag import settings, store
from rag.chunking import chunk_text

log = logging.getLogger("rag.index")
SUMMARY_MODEL_NAME = "qwen2.5-1.5b-qlora-scitldr"


def _document_text(article):
    parts = [article["title"].strip(), article["content"]]
    return "\n\n".join(part for part in parts if part) or article["source_id"]


def embed_pending(conn, embedder, batch_size=64):
    done = 0
    while articles := store.articles_needing_embedding(conn, batch_size):
        chunked = [chunk_text(_document_text(a), embedder.tokenizer) for a in articles]
        vectors = iter(embedder.encode([text for chunks in chunked for text in chunks]))
        for article, chunks in zip(articles, chunked):
            store.save_chunks(conn, article, chunks, [next(vectors) for _ in chunks])
        done += len(articles)
        log.info("embedded %d articles", done)
    return done


def summarize_pending(conn, summarize, batch_size=16):
    done = 0
    while articles := store.articles_needing_summary(conn, batch_size):
        for article in articles:
            store.save_summary(conn, article, summarize(article["content"]), SUMMARY_MODEL_NAME)
        done += len(articles)
        log.info("summarized %d papers", done)
    return done


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job", choices=["embed", "summarize"])
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with psycopg.connect(settings.database_url(), autocommit=True) as conn:
        store.init_schema(conn)
        if args.job == "embed":
            from rag.embedder import Embedder
            print("embedded articles:", embed_pending(conn, Embedder()))
        else:
            from rag.summarizer import Summarizer
            print("summarized papers:", summarize_pending(conn, Summarizer()))


if __name__ == "__main__":
    main()
