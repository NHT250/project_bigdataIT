"""Entry point: one independent polling thread per source."""
import functools
import logging
import os
import signal
import threading

import psycopg

from . import arxiv, config, devto, enrich, hn, store
from .net import Fetcher, make_session
from .pages import PageReader
from .runner import run_forever

# The archive is complete, so polls are spread out to keep the machine light. No source is
# switched off: the Feed still shows new articles arriving. The request limits live with each
# source: arXiv one request every 3 s, Dev.to at most 5 a second, linked pages 5 s apart per site.
POLL_INTERVALS = {"arxiv": 3600, "hn": 1200, "devto": 900}
# Each run reads a batch of story pages, so the backlog clears in a few hours.
CONTENT_INTERVAL_SECONDS = 60


def _job(poll, fetcher, database_url):
    def run():
        with psycopg.connect(database_url, autocommit=True, connect_timeout=10) as conn:
            return poll(fetcher, conn)

    return run


def main():
    settings = config.load(os.environ)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    with psycopg.connect(settings.database_url, autocommit=True, connect_timeout=10) as conn:
        store.init_schema(conn)

    stop = threading.Event()
    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, lambda *_: stop.set())

    sources = {
        "arxiv": (arxiv.poll, arxiv.make_fetcher(make_session(settings.contact_email))),
        "hn": (hn.poll, Fetcher(make_session(settings.contact_email))),
        # Articles without a body are remembered between polls so they aren't fetched again.
        "devto": (functools.partial(devto.poll, skipped=set(), workers=devto.DETAIL_WORKERS),
                  devto.make_fetcher(devto.make_session(settings.contact_email))),
    }
    threads = [
        threading.Thread(
            target=run_forever,
            args=(name, _job(poll, fetcher, settings.database_url), POLL_INTERVALS[name], stop),
            name=name,
        )
        for name, (poll, fetcher) in sources.items()
    ]
    # The page reader keeps robots.txt and per-site timing between runs, so it is made once.
    reader = PageReader(make_session(settings.contact_email))
    content_poll = lambda fetcher, conn: enrich.poll(fetcher, reader, conn, workers=enrich.DOMAIN_WORKERS)  # noqa: E731
    threads.append(threading.Thread(
        target=run_forever,
        args=("hn-content", _job(content_poll, Fetcher(make_session(settings.contact_email)), settings.database_url),
              CONTENT_INTERVAL_SECONDS, stop),
        name="hn-content",
    ))
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
