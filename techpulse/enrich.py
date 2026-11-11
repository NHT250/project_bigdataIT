"""Give Hacker News stories a body for search and answers.

A story's body is built from, in order: the post's own text (Ask HN and the like,
from the official API), the main text of the page it links to (read politely, see
pages.py), and its top comments (from the official API). Each part is labelled so
an answer can tell the article apart from what commenters said.
"""
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from urllib.parse import urlsplit

from . import store
from .net import FetchError
from .text import html_to_text

BASE_URL = "https://hacker-news.firebaseio.com/v0"
# Comments need time to arrive, and new stories are still being voted on.
MIN_AGE = timedelta(hours=1)
STORIES_PER_POLL = 40
MAX_COMMENTS = 5
MAX_COMMENT_LOOKUPS = 10
MAX_COMMENT_CHARS = 1500
MAX_CONTENT_CHARS = 30_000
ITEM_ATTEMPTS = 2
# Sites read at once. Stories are grouped by the site they link to and each group is read
# by one worker in turn, so the 5-second gap per site (pages.py) holds at any setting. One at
# a time keeps the machine light now the backlog is done; up to 12 at once has been tested.
DOMAIN_WORKERS = 1


def _item(fetcher, base_url, item_id):
    return fetcher.get(f"{base_url}/item/{item_id}.json", max_attempts=ITEM_ATTEMPTS).json()


def _comments(fetcher, base_url, kids):
    comments = []
    for kid in kids[:MAX_COMMENT_LOOKUPS]:
        try:
            item = _item(fetcher, base_url, kid)
        except FetchError:
            continue
        if not item or item.get("deleted") or item.get("dead"):
            continue
        text = html_to_text(item.get("text"))
        if text:
            comments.append(text[:MAX_COMMENT_CHARS])
        if len(comments) == MAX_COMMENTS:
            break
    return comments


def _links_elsewhere(url):
    host = urlsplit(url or "").hostname or ""
    return bool(url) and host != "news.ycombinator.com"


def build_content(post_text, page_text, comments):
    parts = []
    if post_text:
        parts.append(f"Post text:\n{post_text}")
    if page_text:
        parts.append(f"Linked page:\n{page_text}")
    if comments:
        parts.append("Top Hacker News comments:\n" + "\n".join(f"- {c}" for c in comments))
    return "\n\n".join(parts)[:MAX_CONTENT_CHARS]


def gather_story(fetcher, reader, base_url, source_id):
    """The story's body, page outcome and comment count, read from the network only."""
    item = _item(fetcher, base_url, source_id) or {}
    post_text = html_to_text(item.get("text"))
    page = reader.read(item["url"]) if _links_elsewhere(item.get("url")) else None
    comments = _comments(fetcher, base_url, item.get("kids") or [])
    page_text = page.text if page and page.status == "ok" else ""
    content = build_content(post_text, page_text, comments)
    return content or None, page.status if page else "none", len(comments)


def enrich_story(fetcher, reader, conn, base_url, source_id):
    content, page_status, comments = gather_story(fetcher, reader, base_url, source_id)
    store.save_story_content(conn, source_id, content, page_status, comments)
    return bool(content)


def _site_of(url):
    return (urlsplit(url or "").hostname or "").lower()


def _read_site(fetcher, reader, base_url, source_ids):
    """One worker's share: the stories linking to one site, one after another."""
    results = []
    for source_id in source_ids:
        try:
            results.append((source_id, gather_story(fetcher, reader, base_url, source_id)))
        except FetchError:
            results.append((source_id, None))  # The story itself couldn't be fetched; try again next poll.
    return results


def poll(fetcher, reader, conn, base_url=BASE_URL, limit=STORIES_PER_POLL, now=None, workers=1):
    """Workers only read from the network; every database write happens on this thread."""
    now = now or datetime.now(timezone.utc)
    ids = store.stories_needing_content(conn, now - MIN_AGE, limit)
    stats = {"stories": 0, "with_content": 0, "failed": 0}
    if not ids:
        return stats
    urls = store.article_urls(conn, "hn", ids)
    by_site = defaultdict(list)
    for source_id in ids:
        by_site[_site_of(urls.get(source_id))].append(source_id)

    def save(results):
        for source_id, gathered in results:
            if gathered is None:
                stats["failed"] += 1
                continue
            content, page_status, comments = gathered
            store.save_story_content(conn, source_id, content, page_status, comments)
            stats["stories"] += 1
            stats["with_content"] += bool(content)

    if workers <= 1:
        for group in by_site.values():
            save(_read_site(fetcher, reader, base_url, group))
        return stats
    with ThreadPoolExecutor(min(workers, len(by_site))) as executor:
        futures = [executor.submit(_read_site, fetcher, reader, base_url, group) for group in by_site.values()]
        for future in as_completed(futures):
            save(future.result())
    return stats
