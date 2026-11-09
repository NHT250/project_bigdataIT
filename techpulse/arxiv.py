"""arXiv API source (Atom feed, at most one request every 3 seconds)."""
import re
import time
from datetime import datetime
from xml.etree import ElementTree

from . import store
from .net import Fetcher, RateLimiter
from .store import Article

API_URL = "http://export.arxiv.org/api/query"
CATEGORIES = ("cs.AI", "cs.LG", "cs.CL", "cs.CR")
MIN_INTERVAL_SECONDS = 3.0
PAGE_SIZE = 100
# Without a previous poll there is no "new since last time", so the first run
# takes the most recent papers up to this many.
BACKFILL_TARGET = 1500
MAX_ITEMS_PER_POLL = 2000

NS = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivApiError(Exception):
    """The API answered 200 with an error entry instead of papers."""


def make_fetcher(session, clock=time.monotonic, sleep=time.sleep, base_delay=5.0):
    limiter = RateLimiter(MIN_INTERVAL_SECONDS, clock=clock, sleep=sleep)
    return Fetcher(session, limiter=limiter, sleep=sleep, base_delay=base_delay)


def _text(entry, tag):
    return " ".join(entry.findtext(f"atom:{tag}", "", NS).split())


def _source_id(entry_id):
    return re.sub(r"v\d+$", "", entry_id.split("/abs/", 1)[1])


def _link(entry, **attributes):
    for link in entry.findall("atom:link", NS):
        if all(link.get(key) == value for key, value in attributes.items()):
            return link.get("href")
    return None


def _parse_entry(entry):
    entry_id = _text(entry, "id")
    if "/api/errors" in entry_id:
        raise ArxivApiError(_text(entry, "summary"))
    names = (a.findtext("atom:name", "", NS).strip() for a in entry.findall("atom:author", NS))
    updated = _text(entry, "updated")
    return Article(
        source="arxiv",
        source_id=_source_id(entry_id),
        title=_text(entry, "title"),
        content=_text(entry, "summary"),
        url=_link(entry, rel="alternate") or entry_id,
        author=", ".join(names),
        categories=[c.get("term") for c in entry.findall("atom:category", NS)],
        published_at=datetime.fromisoformat(_text(entry, "published")),
        extra={"updated": updated, "pdf_url": _link(entry, title="pdf")},
    )


def parse_feed(xml):
    return [_parse_entry(e) for e in ElementTree.fromstring(xml).findall("atom:entry", NS)]


def fetch_page(fetcher, start, page_size):
    response = fetcher.get(API_URL, params={
        "search_query": " OR ".join(f"cat:{c}" for c in CATEGORIES),
        "sortBy": "submittedDate",
        "sortOrder": "descending",
        "start": start,
        "max_results": page_size,
    })
    return parse_feed(response.content)


def poll(fetcher, conn, page_size=PAGE_SIZE, backfill_target=BACKFILL_TARGET):
    """Page newest-first until a page holds nothing new, or the backfill is full."""
    backfilling = store.count(conn, "arxiv") < backfill_target
    limit = backfill_target if backfilling else MAX_ITEMS_PER_POLL
    start = fetched = new = 0
    while start < limit:
        page = fetch_page(fetcher, start, page_size)
        if not page:
            break
        inserted = store.upsert(conn, page)
        fetched += len(page)
        new += inserted
        if len(page) < page_size or (inserted == 0 and not backfilling):
            break
        start += len(page)
    return {"fetched": fetched, "new": new}
