"""Entry point: one independent polling thread per source."""
import logging
import os
import signal
import threading

import psycopg

from . import arxiv, config, hn, store
from .net import Fetcher, make_session
from .runner import run_forever

POLL_INTERVAL_SECONDS = 300


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
    }
    threads = [
        threading.Thread(
            target=run_forever,
            args=(name, _job(poll, fetcher, settings.database_url), POLL_INTERVAL_SECONDS, stop),
            name=name,
        )
        for name, (poll, fetcher) in sources.items()
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
