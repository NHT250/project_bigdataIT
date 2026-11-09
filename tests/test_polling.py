"""Polling again must update rows in place, against a real Postgres."""
import json

from fakes import FakeClock, FakeResponse, FakeSession, arxiv_entry, arxiv_feed
from techpulse import arxiv, hn, store
from techpulse.net import Fetcher


def row_count(conn, source):
    return conn.execute(
        "SELECT count(*) FROM articles WHERE source = %s", (source,)
    ).fetchone()[0]


def arxiv_fetcher(session):
    clock = FakeClock()
    session.clock = clock
    return arxiv.make_fetcher(session, clock=clock.monotonic, sleep=clock.sleep)


def paged_feed(ids_by_start):
    def handler(url, params):
        ids = ids_by_start.get(params["start"], [])
        return FakeResponse(arxiv_feed(*[arxiv_entry(i) for i in ids]))

    return handler


def hn_handler(items, top=None, new=None):
    def handler(url, params):
        if url.endswith("topstories.json"):
            return FakeResponse(json.dumps(sorted(items) if top is None else top))
        if url.endswith("newstories.json"):
            return FakeResponse(json.dumps([] if new is None else new))
        item_id = int(url.rsplit("/", 1)[1].removesuffix(".json"))
        return FakeResponse(json.dumps(items[item_id]))

    return handler


def story(item_id, score=10, **overrides):
    return {
        "id": item_id, "type": "story", "title": f"Story {item_id}",
        "url": f"https://example.com/{item_id}", "score": score,
        "descendants": 3, "time": 1_800_000_000, "by": "pg", **overrides,
    }


def test_arxiv_repoll_creates_no_duplicates(conn):
    session = FakeSession(paged_feed({0: ["2411.00001v1", "2411.00002v1"]}))
    fetcher = arxiv_fetcher(session)
    arxiv.poll(fetcher, conn, backfill_target=0)
    arxiv.poll(fetcher, conn, backfill_target=0)
    assert row_count(conn, "arxiv") == 2


def test_arxiv_new_version_updates_the_existing_row(conn):
    session = FakeSession(paged_feed({0: ["2411.00001v1"]}))
    arxiv.poll(arxiv_fetcher(session), conn, backfill_target=0)
    session = FakeSession(paged_feed({0: ["2411.00001v2"]}))
    arxiv.poll(arxiv_fetcher(session), conn, backfill_target=0)
    assert row_count(conn, "arxiv") == 1
    url = conn.execute("SELECT url FROM articles").fetchone()[0]
    assert url.endswith("v2")


def test_arxiv_query_covers_the_four_categories_newest_first(conn):
    session = FakeSession(paged_feed({}))
    arxiv.poll(arxiv_fetcher(session), conn, backfill_target=0)
    params = session.calls[0][1]
    for category in ("cs.AI", "cs.LG", "cs.CL", "cs.CR"):
        assert f"cat:{category}" in params["search_query"]
    assert params["sortBy"] == "submittedDate"
    assert params["sortOrder"] == "descending"


def test_arxiv_stops_paging_at_a_page_with_nothing_new(conn):
    pages = {0: ["2411.00002v1", "2411.00001v1"], 2: ["2411.00000v1"]}
    fetcher_session = FakeSession(paged_feed(pages))
    fetcher = arxiv_fetcher(fetcher_session)
    store.upsert(conn, arxiv.fetch_page(fetcher, 0, 2))
    fetcher_session.calls.clear()

    arxiv.poll(fetcher, conn, page_size=2, backfill_target=0)

    assert len(fetcher_session.calls) == 1


def test_arxiv_keeps_paging_while_new_papers_keep_appearing(conn):
    pages = {0: ["2411.00004v1", "2411.00003v1"], 2: ["2411.00002v1", "2411.00001v1"]}
    session = FakeSession(paged_feed(pages))

    arxiv.poll(arxiv_fetcher(session), conn, page_size=2, backfill_target=0)

    assert row_count(conn, "arxiv") == 4


def test_arxiv_first_run_backfills_up_to_the_target_and_no_further(conn):
    pages = {
        start: [f"2411.{start + offset:05d}v1" for offset in range(2)]
        for start in range(0, 20, 2)
    }
    session = FakeSession(paged_feed(pages))

    arxiv.poll(arxiv_fetcher(session), conn, page_size=2, backfill_target=6)

    assert row_count(conn, "arxiv") == 6
    assert len(session.calls) == 3


def test_arxiv_interrupted_backfill_resumes_past_known_pages(conn):
    pages = {
        start: [f"2411.{start + offset:05d}v1" for offset in range(2)]
        for start in range(0, 20, 2)
    }
    fetcher = arxiv_fetcher(FakeSession(paged_feed(pages)))
    store.upsert(conn, arxiv.fetch_page(fetcher, 0, 2))

    arxiv.poll(fetcher, conn, page_size=2, backfill_target=6)

    assert row_count(conn, "arxiv") == 6


def test_hn_repoll_creates_no_duplicates(conn):
    fetcher = Fetcher(FakeSession(hn_handler({1: story(1), 2: story(2)})))
    hn.poll(fetcher, conn)
    hn.poll(fetcher, conn)
    assert row_count(conn, "hn") == 2


def test_hn_story_in_both_lists_is_stored_once(conn):
    items = {1: story(1)}
    fetcher = Fetcher(FakeSession(hn_handler(items, top=[1], new=[1])))
    hn.poll(fetcher, conn)
    assert row_count(conn, "hn") == 1


def test_hn_repoll_updates_score_and_comment_count(conn):
    hn.poll(Fetcher(FakeSession(hn_handler({1: story(1, score=10)}))), conn)
    updated = story(1, score=99, descendants=42)
    hn.poll(Fetcher(FakeSession(hn_handler({1: updated}))), conn)
    score, extra = conn.execute("SELECT score, extra FROM articles").fetchone()
    assert score == 99
    assert extra["descendants"] == 42


def test_repoll_refreshes_fetched_at(conn):
    fetcher = Fetcher(FakeSession(hn_handler({1: story(1)})))
    hn.poll(fetcher, conn)
    first = conn.execute("SELECT fetched_at FROM articles").fetchone()[0]
    hn.poll(fetcher, conn)
    second = conn.execute("SELECT fetched_at FROM articles").fetchone()[0]
    assert second > first


def test_hn_stored_fields_map_to_the_articles_table(conn):
    hn.poll(Fetcher(FakeSession(hn_handler({1: story(1, score=7)}))), conn)
    row = conn.execute(
        "SELECT source, source_id, title, url, author, score FROM articles"
    ).fetchone()
    assert row == ("hn", "1", "Story 1", "https://example.com/1", "pg", 7)


def test_hn_stores_only_stories(conn):
    items = {
        1: story(1),
        2: story(2, type="job"),
        3: story(3, type="poll"),
        4: story(4, type="comment"),
    }
    hn.poll(Fetcher(FakeSession(hn_handler(items))), conn)
    assert row_count(conn, "hn") == 1


def test_hn_skips_deleted_dead_and_missing_items(conn):
    items = {
        1: story(1),
        2: {"id": 2, "type": "story", "deleted": True, "time": 1},
        3: story(3, dead=True),
        4: None,
    }
    hn.poll(Fetcher(FakeSession(hn_handler(items))), conn)
    assert row_count(conn, "hn") == 1


def test_hn_story_without_url_links_to_its_discussion(conn):
    item = story(1)
    del item["url"]
    hn.poll(Fetcher(FakeSession(hn_handler({1: item}))), conn)
    url = conn.execute("SELECT url FROM articles").fetchone()[0]
    assert url == "https://news.ycombinator.com/item?id=1"
