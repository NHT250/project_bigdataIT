"""Ask a question about the collected articles from the command line."""
import argparse
import sys

from rag import answering, settings

TOP_K = 5


def run(question, api_key, embedder=None, history=()):
    """Heavy imports stay here so a missing key is reported immediately."""
    import psycopg
    from google import genai

    from rag import store
    from rag.embedder import Embedder

    embedder = embedder or Embedder()
    generate = answering.GeminiGenerator(
        genai.Client(api_key=api_key),
        model=settings.get("GEMINI_MODEL", answering.DEFAULT_MODEL),
    )
    with psycopg.connect(settings.database_url(), autocommit=True) as conn:
        return answering.answer(
            question,
            retrieve=lambda q: store.search(conn, embedder.encode([q])[0], TOP_K),
            generate=generate,
            searched_articles=store.indexed_article_count(conn),
            api_key=api_key,
            history=history,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        api_key = settings.get("GEMINI_API_KEY")
        answering.require_api_key(api_key)
        result = run(args.question, api_key)
    except (answering.MissingApiKey, answering.AnswerError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 2

    print(f"(searched {result.searched_articles} articles)\n\n{result.text}\n")
    for source in result.sources:
        print(f"[{source.number}] {source.title}\n    {source.url}")
        if source.summary:
            print(f"    Summary: {source.summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
