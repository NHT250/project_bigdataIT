"""Hacker News official Firebase API source (stories only)."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from . import store
from .net import FetchError
from .store import Article

BASE_URL = "https://hacker-news.firebaseio.com/v0"
LISTS = ("topstories", "newstories")
WORKERS = 8
MAX_ITEM_FAILURES = 50
ITEM_ATTEMPTS = 2


def to_article(item):
    """Map an API item to an Article, or None if it is not a live story."""
    if not item or item.get("type") != "story":
        return None
    if item.get("deleted") or item.get("dead") or not item.get("title"):
        return None
    item_id = item["id"]
    return Article(
        source="hn",
        source_id=str(item_id),
        title=item["title"],
        content=None,
        url=item.get("url") or f"https://news.ycombinator.com/item?id={item_id}",
        author=item.get("by"),
        categories=[],
        published_at=datetime.fromtimestamp(item["time"], tz=timezone.utc),
        score=item.get("score"),
        extra={"descendants": item.get("descendants")},
    )


def _story_ids(fetcher, base_url):
    ids = []
    for name in LISTS:
        ids.extend(fetcher.get(f"{base_url}/{name}.json").json())
    return list(dict.fromkeys(ids))


def _fetch_item(fetcher, base_url, item_id):
    url = f"{base_url}/item/{item_id}.json"
    return fetcher.get(url, max_attempts=ITEM_ATTEMPTS).json()


def poll(fetcher, conn, base_url=BASE_URL):
    """Fetch both lists. A few bad items are skipped; many mean the network is down."""
    ids = _story_ids(fetcher, base_url)
    articles = []
    failed = 0
    with ThreadPoolExecutor(WORKERS) as executor:
        futures = [executor.submit(_fetch_item, fetcher, base_url, i) for i in ids]
        for future in as_completed(futures):
            try:
                article = to_article(future.result())
            except FetchError:
                failed += 1
                if failed > MAX_ITEM_FAILURES:
                    for pending in futures:
                        pending.cancel()
                    raise FetchError(f"{failed} Hacker News items failed; aborting poll")
                continue
            if article is not None:
                articles.append(article)
    new = store.upsert(conn, articles)
    return {"stored": len(articles), "new": new, "failed": failed}
