"""Dev.to source: parsing, re-polling, rate limit, User-Agent and failures."""
from datetime import datetime, timedelta, timezone

import requests

from fakes import FakeClock, FakeResponse, FakeSession
from techpulse import devto
from techpulse.net import build_user_agent, make_session

NOW = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
BASE = "https://dev.to/api"
EMAIL = "student@example.invalid"


def listed(item_id, hours_ago=1, reactions=5, comments=2, tags=("python", "ai")):
    published = (NOW - timedelta(hours=hours_ago)).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "id": item_id, "title": f"Post {item_id}", "description": f"About post {item_id}",
        "url": f"https://dev.to/grace/post-{item_id}", "published_at": published,
        "tag_list": list(tags), "public_reactions_count": reactions, "comments_count": comments,
        "reading_time_minutes": 4, "user": {"name": "Grace Hopper", "username": "grace"},
    }


def detail(item_id, body="# Setup\n\nInstall the package, then run it."):
    article = {"id": item_id, "title": f"Post {item_id}", "tags": ["python", "ai"]}
    if body is not ...:
        article["body_markdown"] = body
    return article


class Api:
    """Dev.to API double: pages per tag, and one detail (or error) per id."""

    def __init__(self, pages, details):
        self.pages = pages
        self.details = details

    def __call__(self, url, params):
        if url == f"{BASE}/articles":
            tag_pages = self.pages.get(params["tag"], [])
            index = params["page"] - 1
            return FakeResponse(_json(tag_pages[index] if index < len(tag_pages) else []))
        item_id = int(url.rsplit("/", 1)[1])
        outcome = self.details[item_id]
        if isinstance(outcome, list):  # a scripted sequence of outcomes, one per call
            outcome = outcome.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        if isinstance(outcome, FakeResponse):
            return outcome
        return FakeResponse(_json(outcome))


def _json(value):
    import json

    return json.dumps(value)


def fetcher_for(api):
    clock = FakeClock()
    session = FakeSession(api, clock)
    return devto.make_fetcher(session, clock=clock.monotonic, sleep=clock.sleep), session, clock


def rows(conn):
    return conn.execute(
        "SELECT source_id, title, content, url, author, categories, score, extra"
        " FROM articles WHERE source = 'devto' ORDER BY source_id"
    ).fetchall()


def detail_calls(session):
    return [url for url, _ in session.calls if url != f"{BASE}/articles"]


# Parsing


def test_listing_and_detail_become_one_article():
    article = devto.to_article(listed(7, reactions=12, comments=3), detail(7, body="# Hi\n\nText."))
    assert article.source == "devto"
    assert article.source_id == "7"
    assert article.title == "Post 7"
    assert article.content == "# Hi\n\nText."
    assert article.url == "https://dev.to/grace/post-7"
    assert article.author == "Grace Hopper"
    assert article.categories == ["python", "ai"]
    assert article.published_at == NOW - timedelta(hours=1)
    assert article.score == 12
    assert article.extra["comments"] == 3
    assert article.extra["description"] == "About post 7"


def test_author_falls_back_to_the_username():
    item = listed(7)
    item["user"] = {"name": "", "username": "grace"}
    assert devto.to_article(item, detail(7)).author == "grace"


def test_article_without_body_markdown_is_dropped():
    assert devto.to_article(listed(7), detail(7, body=...)) is None
    assert devto.to_article(listed(7), detail(7, body=None)) is None
    assert devto.to_article(listed(7), detail(7, body="   \n")) is None


# Polling against a real Postgres


def test_poll_stores_each_article_with_its_full_text(conn):
    api = Api({"python": [[listed(1), listed(2)]]}, {1: detail(1), 2: detail(2, body="Second body")})
    fetcher, _, _ = fetcher_for(api)
    stats = devto.poll(fetcher, conn, now=NOW, tags=["python"])
    stored = rows(conn)
    assert [row[0] for row in stored] == ["1", "2"]
    assert stored[1][2] == "Second body"
    assert stats["new"] == 2


def test_repoll_creates_no_duplicates_and_does_not_refetch_bodies(conn):
    api = Api({"python": [[listed(1), listed(2)]]}, {1: detail(1), 2: detail(2)})
    fetcher, session, _ = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    first_details = len(detail_calls(session))
    api.pages = {"python": [[listed(1, reactions=40), listed(2)]]}
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    stored = rows(conn)
    assert len(stored) == 2
    assert len(detail_calls(session)) == first_details
    assert stored[0][6] == 40  # the reaction count is refreshed
    assert stored[0][2].startswith("# Setup")  # the body is kept


def test_an_article_in_two_tags_is_stored_once_and_fetched_once(conn):
    api = Api({"python": [[listed(1)]], "ai": [[listed(1), listed(2)]]}, {1: detail(1), 2: detail(2)})
    fetcher, session, _ = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python", "ai"])
    assert [row[0] for row in rows(conn)] == ["1", "2"]
    assert detail_calls(session).count(f"{BASE}/articles/1") == 1


def test_articles_older_than_seven_days_are_left_out_and_paging_stops(conn):
    pages = {"python": [[listed(1), listed(2, hours_ago=24 * 8)], [listed(3, hours_ago=24 * 9)], [listed(4)]]}
    api = Api(pages, {1: detail(1), 2: detail(2), 3: detail(3), 4: detail(4)})
    fetcher, session, _ = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    assert [row[0] for row in rows(conn)] == ["1"]
    pages_read = [params["page"] for url, params in session.calls if url == f"{BASE}/articles"]
    assert pages_read == [1, 2]


def test_paging_stops_at_a_page_with_nothing_new(conn):
    api = Api({"python": [[listed(1)], [listed(2)]]}, {1: detail(1), 2: detail(2)})
    fetcher, session, _ = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    session.calls.clear()
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    pages_read = [params["page"] for url, params in session.calls if url == f"{BASE}/articles"]
    assert pages_read == [1]


def test_article_without_body_is_skipped_and_the_rest_are_stored(conn):
    api = Api({"python": [[listed(1), listed(2), listed(3)]]},
              {1: detail(1), 2: detail(2, body=...), 3: detail(3, body="")})
    fetcher, _, _ = fetcher_for(api)
    stats = devto.poll(fetcher, conn, now=NOW, tags=["python"])
    assert [row[0] for row in rows(conn)] == ["1"]
    assert stats["no_body"] == 2


def test_skipped_articles_are_not_fetched_again_next_poll(conn):
    api = Api({"python": [[listed(1), listed(2)]]}, {1: detail(1), 2: detail(2, body=...)})
    fetcher, session, _ = fetcher_for(api)
    skipped = set()
    devto.poll(fetcher, conn, now=NOW, tags=["python"], skipped=skipped)
    devto.poll(fetcher, conn, now=NOW, tags=["python"], skipped=skipped)
    assert detail_calls(session).count(f"{BASE}/articles/2") == 1


# Rate limit and User-Agent


def test_requests_never_exceed_five_per_second(conn):
    api = Api({"python": [[listed(i) for i in range(1, 13)]]}, {i: detail(i) for i in range(1, 13)})
    fetcher, session, _ = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    times = session.call_times
    assert len(times) >= 13
    assert all(later - earlier >= 0.2 - 1e-9 for earlier, later in zip(times, times[1:]))
    for start in range(len(times) - 5):
        assert times[start + 5] - times[start] >= 1.0 - 1e-9


def test_dev_to_requests_carry_the_honest_user_agent():
    # Dev.to's firewall answers 403 to the long project name, so its requests spell it VLU.
    user_agent = devto.make_session(EMAIL).headers["User-Agent"]
    assert user_agent == f"TechPulse/1.0 (VLU course project; {EMAIL})"
    assert "Mozilla" not in user_agent


def test_the_other_sources_keep_the_full_project_name():
    assert build_user_agent(EMAIL) == f"TechPulse/1.0 (Van Lang University course project; {EMAIL})"
    assert make_session(EMAIL).headers["User-Agent"] == build_user_agent(EMAIL)


# Network failures


def test_network_error_backs_off_and_then_succeeds(conn):
    flaky = [requests.ConnectionError("reset"), FakeResponse("", status_code=503), detail(1)]
    api = Api({"python": [[listed(1)]]}, {1: flaky})
    fetcher, _, clock = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    assert [row[0] for row in rows(conn)] == ["1"]
    backoffs = [s for s in clock.sleeps if s >= 1]
    assert len(backoffs) == 2
    assert backoffs[1] > backoffs[0]


def test_an_article_that_keeps_failing_does_not_stop_the_job(conn):
    always_down = [requests.ConnectionError("reset")] * 10
    api = Api({"python": [[listed(1), listed(2)]]}, {1: always_down, 2: detail(2)})
    fetcher, _, _ = fetcher_for(api)
    stats = devto.poll(fetcher, conn, now=NOW, tags=["python"])
    assert [row[0] for row in rows(conn)] == ["2"]
    assert stats["failed"] == 1


def test_a_failed_article_is_tried_again_next_poll(conn):
    api = Api({"python": [[listed(1)]]}, {1: [requests.ConnectionError("reset")] * 5 + [detail(1)]})
    fetcher, _, _ = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    assert rows(conn) == []
    devto.poll(fetcher, conn, now=NOW, tags=["python"])
    assert [row[0] for row in rows(conn)] == ["1"]


# Schema


def test_a_database_made_before_dev_to_accepts_dev_to_rows(conn):
    from techpulse import store

    conn.execute("ALTER TABLE articles DROP CONSTRAINT articles_source_check")
    conn.execute("ALTER TABLE articles ADD CONSTRAINT articles_source_check CHECK (source IN ('arxiv', 'hn'))")
    store.init_schema(conn)
    store.upsert(conn, [devto.to_article(listed(1), detail(1))])
    assert [row[0] for row in rows(conn)] == ["1"]
