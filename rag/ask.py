"""Ask a question about the collected articles from the command line."""
import argparse
import sys

from rag import answering, providers, retrieval, settings, suggestions

TOP_K = 5


def retrieval_mode(get):
    """hybrid (default), dense or bm25: see rag.retrieval."""
    mode = (get("RETRIEVAL_MODE") or "hybrid").strip().lower()
    if mode not in retrieval.MODES:
        raise ValueError(f'RETRIEVAL_MODE must be one of {", ".join(retrieval.MODES)}, not "{mode}".')
    return mode


def _retriever(conn, embedder):
    mode = retrieval_mode(settings.get)
    return lambda q: retrieval.retrieve(conn, q, embedder.encode([q])[0], mode, TOP_K)


# answering.answer checks that a key is set; build_generator has already made sure of it.
KEY_CHECKED = "checked by providers.build_generator"


def run(question, generator=None, embedder=None, history=()):
    """Heavy imports stay here so a missing key is reported immediately."""
    import psycopg

    from rag import store
    from rag.embedder import Embedder

    generator = generator or providers.build_generator(settings.get)
    embedder = embedder or Embedder()
    with psycopg.connect(settings.database_url(), autocommit=True) as conn:
        return answering.answer(
            question,
            retrieve=_retriever(conn, embedder),
            generate=generator,
            searched_articles=store.indexed_article_count(conn),
            api_key=KEY_CHECKED,
            history=history,
        )


# Shared by every question the website answers, so a question asked again isn't sent twice.
SUGGESTION_CACHE = suggestions.Cache()


def run_stream(question, generator, embedder, history=()):
    """The events of answering.answer_events, with the database open until the last one."""
    import psycopg

    from rag import store

    with psycopg.connect(settings.database_url(), autocommit=True) as conn:
        retrieve = _retriever(conn, embedder)
        yield from answering.answer_events(
            question,
            retrieve=retrieve,
            generate=generator,
            stream=generator.stream,
            suggest=suggestions.Suggester(generator, retrieve, cache=SUGGESTION_CACHE),
            searched_articles=store.indexed_article_count(conn),
            api_key=KEY_CHECKED,
            history=history,
        )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")

    try:
        result = run(args.question, providers.build_generator(settings.get))
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
