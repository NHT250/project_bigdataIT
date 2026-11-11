"""Dev.to public API source: practical articles, with their full Markdown text.

The tag listing has no body, so each new article costs one more request for it.
Every request, retries included, goes through one limiter: at most 5 per second.
"""
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from . import store
from .net import FetchError, Fetcher, RateLimiter
from .net import make_session as _make_session
from .store import Article

BASE_URL = "https://dev.to/api"
TAGS = ("ai", "machinelearning", "python", "webdev", "programming", "datascience")
MIN_INTERVAL_SECONDS = 0.2  # 5 requests per second
PAGE_SIZE = 100
MAX_AGE = timedelta(days=7)
# The listing is roughly newest first; this only guards against paging forever. The busiest
# tag held about 10 days in 60 pages of 100 (11/2026).
MAX_PAGES_PER_TAG = 100
# Bodies in flight at once. One at a time keeps to about 1.4 requests a second, which ran for
# hours without a single 429; eight at once, nearer the 5 a second ceiling, drew 429s from
# Dev.to within minutes (11/2026). Dev.to publishes no number, so its answers decide.
DETAIL_WORKERS = 1
# A few articles that won't load are skipped; this many in one poll means the network is down.
MAX_BODY_FAILURES = 25


# Dev.to's firewall answers 403 to a User-Agent naming the project in full (checked
# 11/2026), so its requests spell the university VLU. It is still the same honest
# name with the contact email, never a browser's.
PROJECT = "VLU course project"


def make_session(email):
    return _make_session(email, PROJECT)


def make_fetcher(session, clock=time.monotonic, sleep=time.sleep, base_delay=2.0):
    limiter = RateLimiter(MIN_INTERVAL_SECONDS, clock=clock, sleep=sleep)
    return Fetcher(session, limiter=limiter, sleep=sleep, base_delay=base_delay)


def _published(item):
    return datetime.fromisoformat(item["published_at"].replace("Z", "+00:00"))


def _author(item):
    user = item.get("user") or {}
    return (user.get("name") or "").strip() or user.get("username")


def _listed_article(item, content):
    tags = item.get("tag_list")
    return Article(
        source="devto",
        source_id=str(item["id"]),
        title=item["title"],
        content=content,
        url=item.get("url"),
        author=_author(item),
        categories=list(tags) if isinstance(tags, list) else [],
        published_at=_published(item),
        score=item.get("public_reactions_count"),
        extra={
            "description": item.get("description") or "",
            "comments": item.get("comments_count"),
            "reading_minutes": item.get("reading_time_minutes"),
        },
    )


def to_article(item, detail):
    """Join a listing entry with its full text, or None when the article has no body."""
    body = (detail or {}).get("body_markdown")
    if not isinstance(body, str) or not body.strip():
        return None
    return _listed_article(item, body)


def _listing(fetcher, base_url, tag, page, page_size):
    response = fetcher.get(f"{base_url}/articles", params={"tag": tag, "per_page": page_size, "page": page})
    return response.json()


def _details(fetcher, base_url, source_ids, workers):
    """Each id's article detail, or the FetchError it failed with, in the order given."""
    def one(source_id):
        try:
            return fetcher.get(f"{base_url}/articles/{source_id}").json()
        except FetchError as error:
            return error

    if workers <= 1 or len(source_ids) <= 1:
        return [one(source_id) for source_id in source_ids]
    with ThreadPoolExecutor(min(workers, len(source_ids))) as executor:
        return list(executor.map(one, source_ids))


def poll(fetcher, conn, base_url=BASE_URL, now=None, tags=TAGS, page_size=PAGE_SIZE, skipped=None, workers=1,
         deep=False):
    """Read each tag newest first until a page holds nothing new or nothing from the last 7 days.

    A deep poll (the manual `python -m techpulse.devto`) reads on past pages with nothing new,
    down to the 7-day mark, so articles older than the last poll reached are taken too.

    `skipped` remembers articles without a body across polls, so they aren't fetched again.
    Articles that failed to load are not remembered, so the next poll tries them again.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - MAX_AGE
    skipped = set() if skipped is None else skipped
    added = set()  # stored during this poll: still news for the other tags' stopping rule
    stats = {"listed": 0, "new": 0, "no_body": 0, "failed": 0}
    for tag in tags:
        for page in range(1, MAX_PAGES_PER_TAG + 1):
            items = _listing(fetcher, base_url, tag, page, page_size)
            recent = [item for item in items if _published(item) >= cutoff]
            stats["listed"] += len(items)
            if not recent:
                break
            ids = [str(item["id"]) for item in recent]
            known = store.existing_ids(conn, "devto", ids)
            fresh_this_page = [i for i in ids if (i not in known or i in added) and i not in skipped]
            refreshed, new = [], []
            wanted = [str(item["id"]) for item in recent
                      if str(item["id"]) not in known and str(item["id"]) not in skipped]
            details = dict(zip(wanted, _details(fetcher, base_url, wanted, workers)))
            for item in recent:
                source_id = str(item["id"])
                if source_id in known:
                    # The listing carries current reaction and comment counts; the body is kept.
                    refreshed.append(_listed_article(item, None))
                    continue
                if source_id in skipped or source_id not in details:
                    continue
                detail = details.pop(source_id)  # an article listed twice on a page is stored once
                if isinstance(detail, FetchError):
                    stats["failed"] += 1
                    if stats["failed"] > MAX_BODY_FAILURES:
                        raise detail
                    continue
                article = to_article(item, detail)
                if article is None:
                    skipped.add(source_id)
                    stats["no_body"] += 1
                    continue
                new.append(article)
                added.add(source_id)
            stats["new"] += store.upsert(conn, refreshed + new)
            if not fresh_this_page and not deep:
                break
    return stats


def main():
    """One poll by hand, e.g. to take the 7-day backfill without restarting the collector."""
    import logging
    import os

    import psycopg

    from . import config
    settings = config.load(os.environ)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    with psycopg.connect(settings.database_url, autocommit=True, connect_timeout=10) as conn:
        store.init_schema(conn)
        print("devto poll:", poll(make_fetcher(make_session(settings.contact_email)), conn,
                                  workers=DETAIL_WORKERS, deep=True))


if __name__ == "__main__":
    main()
