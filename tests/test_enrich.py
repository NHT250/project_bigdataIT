"""Giving Hacker News stories a body: post text, the linked page and top comments."""
import json
from datetime import datetime, timedelta, timezone

from fakes import FakeResponse, FakeSession
from techpulse import enrich, hn, store
from techpulse.net import Fetcher
from techpulse.pages import PageResult
from techpulse.store import Article

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
BASE = "https://hn.test/v0"


class FakeReader:
    def __init__(self, results=None):
        self.results = results or {}
        self.urls = []

    def read(self, url):
        self.urls.append(url)
        return self.results.get(url, PageResult("no_text"))


def story(item_id, hours_old=3, url="https://example.com/post"):
    return Article(
        source="hn", source_id=str(item_id), title=f"Story {item_id}", content=None,
        url=url or f"https://news.ycombinator.com/item?id={item_id}", author="pg", categories=[],
        published_at=NOW - timedelta(hours=hours_old), score=10,
    )


def api(items):
    def handler(url, params):
        item_id = int(url.rsplit("/", 1)[1].removesuffix(".json"))
        if item_id not in items:
            return FakeResponse("null")
        return FakeResponse(json.dumps(items[item_id]))
    session = FakeSession(handler)
    return Fetcher(session, sleep=lambda _: None), session


def comment(item_id, text, **extra):
    return {"id": item_id, "type": "comment", "text": text, **extra}


def run(conn, items, reader=None, **kwargs):
    fetcher, session = api(items)
    reader = reader or FakeReader()
    stats = enrich.poll(fetcher, reader, conn, base_url=BASE, now=NOW, **kwargs)
    return stats, reader, session


def content(conn, item_id):
    return conn.execute(
        "SELECT content FROM articles WHERE source = 'hn' AND source_id = %s", (str(item_id),)
    ).fetchone()[0]


def test_linked_page_and_top_comments_become_the_story_body(conn):
    store.upsert(conn, [story(1)])
    items = {
        1: {"id": 1, "type": "story", "url": "https://example.com/post", "kids": [11, 12]},
        11: comment(11, "Faster than the last one &amp; cheaper."),
        12: comment(12, "Context window is <i>huge</i>."),
    }
    reader = FakeReader({"https://example.com/post": PageResult("ok", "Gemini 4 Carbon is a new model.")})
    run(conn, items, reader)
    body = content(conn, 1)
    assert "Linked page:\nGemini 4 Carbon is a new model." in body
    assert "Top Hacker News comments:\n- Faster than the last one & cheaper.\n- Context window is huge." in body


def test_post_text_of_an_ask_hn_story_is_used_without_reading_a_page(conn):
    store.upsert(conn, [story(2, url=None)])
    items = {2: {"id": 2, "type": "story", "text": "What is your setup?<p>Mine is vim."}}
    _, reader, _ = run(conn, items)
    assert content(conn, 2) == "Post text:\nWhat is your setup?\n\nMine is vim."
    assert reader.urls == []


def test_dead_deleted_and_empty_comments_are_skipped_and_at_most_five_kept(conn):
    store.upsert(conn, [story(3)])
    kids = list(range(30, 40))
    items = {3: {"id": 3, "type": "story", "url": "https://example.com/post", "kids": kids}}
    items[30] = comment(30, "gone", deleted=True)
    items[31] = comment(31, "flagged", dead=True)
    items[32] = {"id": 32, "type": "comment"}
    for n in range(33, 40):
        items[n] = comment(n, f"Comment {n}")
    run(conn, items)
    body = content(conn, 3)
    assert "gone" not in body and "flagged" not in body
    assert [f"Comment {n}" in body for n in range(33, 40)] == [True] * 5 + [False] * 2


def test_story_with_nothing_found_keeps_no_body_but_is_not_tried_again(conn):
    store.upsert(conn, [story(4)])
    items = {4: {"id": 4, "type": "story", "url": "https://example.com/post"}}
    first, reader, _ = run(conn, items)
    assert content(conn, 4) is None
    second, reader_again, _ = run(conn, items)
    assert (first["stories"], second["stories"]) == (1, 0)
    assert reader_again.urls == []


def test_outcome_of_the_page_read_is_recorded(conn):
    store.upsert(conn, [story(5)])
    items = {5: {"id": 5, "type": "story", "url": "https://example.com/post"}}
    run(conn, items, FakeReader({"https://example.com/post": PageResult("robots")}))
    row = conn.execute("SELECT page_status, comments FROM hn_content WHERE source_id = '5'").fetchone()
    assert row == ("robots", 0)


def test_stories_too_new_for_comments_wait(conn):
    store.upsert(conn, [story(6, hours_old=0.2)])
    stats, _, session = run(conn, {6: {"id": 6, "type": "story"}})
    assert stats["stories"] == 0 and session.calls == []


def test_arxiv_papers_are_never_touched(conn):
    paper = Article(source="arxiv", source_id="2611.1", title="Paper", content="Abstract", url="https://arxiv.org/abs/2611.1",
                    author="Ada", categories=[], published_at=NOW - timedelta(days=1))
    store.upsert(conn, [paper])
    stats, _, session = run(conn, {})
    assert stats["stories"] == 0 and session.calls == []


def test_newest_stories_are_done_first_up_to_the_limit(conn):
    store.upsert(conn, [story(7, hours_old=5), story(8, hours_old=2), story(9, hours_old=3)])
    items = {n: {"id": n, "type": "story", "text": f"Body {n}"} for n in (7, 8, 9)}
    run(conn, items, limit=2)
    assert (content(conn, 8), content(conn, 9), content(conn, 7)) == ("Post text:\nBody 8", "Post text:\nBody 9", None)


def test_story_whose_item_cannot_be_fetched_is_retried_later(conn):
    store.upsert(conn, [story(10)])

    def handler(url, params):
        return FakeResponse("busy", status_code=404)

    fetcher = Fetcher(FakeSession(handler), sleep=lambda _: None)
    stats = enrich.poll(fetcher, FakeReader(), conn, base_url=BASE, now=NOW)
    assert stats["failed"] == 1
    assert conn.execute("SELECT count(*) FROM hn_content").fetchone()[0] == 0


def test_polling_hacker_news_again_keeps_the_body(conn):
    store.upsert(conn, [story(11)])
    run(conn, {11: {"id": 11, "type": "story", "text": "Kept body"}})
    store.upsert(conn, [hn.to_article({"id": 11, "type": "story", "title": "Story 11", "time": 1_700_000_000, "score": 50})])
    assert content(conn, 11) == "Post text:\nKept body"


def test_body_is_capped(conn):
    store.upsert(conn, [story(12)])
    items = {12: {"id": 12, "type": "story", "url": "https://example.com/post"}}
    reader = FakeReader({"https://example.com/post": PageResult("ok", "word " * 20000)})
    run(conn, items, reader)
    assert len(content(conn, 12)) <= enrich.MAX_CONTENT_CHARS
