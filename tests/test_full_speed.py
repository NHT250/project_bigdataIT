"""Collecting within each provider's limits, and the parallel readers kept for when speed is wanted.

Real threads and the real clock: these check that work runs in parallel while every
per-provider and per-site limit still holds across threads.
"""
import json
import threading
import time
from datetime import datetime, timedelta, timezone

from fakes import FakeResponse
from techpulse import arxiv, devto, enrich, pages
from techpulse import __main__ as collector
from techpulse.net import Fetcher, RateLimiter
from techpulse.pages import PageReader, PageResult
from techpulse.store import Article

EPSILON = 0.01


def gaps(times):
    times = sorted(times)
    return [later - earlier for earlier, later in zip(times, times[1:])]


# The limits themselves


def test_poll_intervals_are_light_for_the_demo():
    # The archive is complete; polls only keep the Feed moving.
    assert collector.POLL_INTERVALS == {"devto": 900, "hn": 1200, "arxiv": 3600}


def test_the_provider_limits_are_unchanged():
    assert arxiv.MIN_INTERVAL_SECONDS == 3.0  # arXiv: one request every three seconds
    assert devto.MIN_INTERVAL_SECONDS == 0.2  # Dev.to: five requests a second
    assert pages.MIN_INTERVAL_SECONDS == 5.0  # linked pages: five seconds between requests to one site


def test_linked_pages_are_read_one_site_at_a_time():
    assert enrich.DOMAIN_WORKERS == 1


def test_dev_to_keeps_to_the_last_seven_days():
    assert devto.MAX_AGE == timedelta(days=7)


def test_the_collector_never_starts_with_a_deep_dev_to_poll():
    assert not hasattr(collector, "first_poll_deep")


# One limiter shared by several threads


def test_a_shared_limiter_spaces_requests_across_threads():
    limiter = RateLimiter(0.05)
    starts = []
    lock = threading.Lock()

    def worker():
        for _ in range(5):
            limiter.acquire()
            with lock:
                starts.append(time.monotonic())

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(starts) == 20
    assert min(gaps(starts)) >= 0.05 - EPSILON


# Dev.to: several bodies in flight, at most five requests a second


class SlowDevto:
    """Answers like the Dev.to API, taking `latency` seconds per request, and records when each began."""

    def __init__(self, count, latency):
        now = datetime.now(timezone.utc)
        self.listing = [{
            "id": i, "title": f"Post {i}", "description": "", "url": f"https://dev.to/a/{i}",
            "published_at": (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "tag_list": ["python"], "public_reactions_count": 1, "comments_count": 0,
            "reading_time_minutes": 1, "user": {"name": "A", "username": "a"},
        } for i in range(1, count + 1)]
        self.latency = latency
        self.starts = []
        self.lock = threading.Lock()

    def get(self, url, params=None, timeout=None):
        with self.lock:
            self.starts.append(time.monotonic())
        time.sleep(self.latency)
        if url.endswith("/articles"):
            return FakeResponse(json.dumps(self.listing if params["page"] == 1 else []))
        item_id = int(url.rsplit("/", 1)[1])
        return FakeResponse(json.dumps({"id": item_id, "body_markdown": f"Body {item_id}"}))


def test_parallel_bodies_never_pass_five_a_second(conn):
    api = SlowDevto(count=12, latency=0.5)
    fetcher = devto.make_fetcher(api)
    start = time.monotonic()
    stats = devto.poll(fetcher, conn, tags=["python"], workers=4)
    elapsed = time.monotonic() - start
    assert stats["new"] == 12
    assert min(gaps(api.starts)) >= 0.2 - EPSILON
    for first in range(len(api.starts) - 5):
        assert sorted(api.starts)[first + 5] - sorted(api.starts)[first] >= 1.0 - EPSILON
    # One at a time, 14 requests of half a second would take 7 seconds.
    assert elapsed < 5


# Linked pages: one site at a time, many sites at once


ARTICLE = "<html><body><article>" + "<p>" + "Plenty of article text here. " * 20 + "</p>" * 3 + "</article></body></html>"


class PageResponse:
    def __init__(self, body, status_code=200, content_type="text/html"):
        self.content = body.encode()
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        self.encoding = "utf-8"

    @property
    def text(self):
        return self.content.decode()

    def iter_content(self, chunk_size=1):
        yield self.content

    def close(self):
        pass


class SlowWeb:
    """Every site allows everything; each page takes `latency` seconds; request start times are recorded."""

    def __init__(self, latency):
        self.latency = latency
        self.starts = []
        self.lock = threading.Lock()

    def get(self, url, timeout=None, stream=False, allow_redirects=True):
        if url.endswith("/robots.txt"):
            return PageResponse("", status_code=404)
        with self.lock:
            self.starts.append((url, time.monotonic()))
        time.sleep(self.latency)
        return PageResponse(ARTICLE)


def read_together(reader, urls):
    threads = [threading.Thread(target=reader.read, args=(url,)) for url in urls]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


def test_two_threads_never_hit_one_site_closer_than_the_interval():
    web = SlowWeb(latency=0.05)
    reader = PageReader(web, min_interval=0.3)
    read_together(reader, [f"https://same.example/{n}" for n in range(4)])
    assert len(web.starts) == 4
    assert min(gaps([when for _, when in web.starts])) >= 0.3 - EPSILON


def test_different_sites_are_read_at_the_same_time():
    web = SlowWeb(latency=0.3)
    reader = PageReader(web, min_interval=5.0)
    read_together(reader, [f"https://site{n}.example/a" for n in range(6)])
    starts = sorted(when for _, when in web.starts)
    # One after another, the last would start at least 1.5 seconds after the first.
    assert starts[-1] - starts[0] < 0.3


def test_robots_txt_is_still_obeyed_by_parallel_readers():
    class Forbidding(SlowWeb):
        def get(self, url, timeout=None, stream=False, allow_redirects=True):
            if url.endswith("/robots.txt"):
                return PageResponse("User-agent: *\nDisallow: /\n", content_type="text/plain")
            return super().get(url, timeout, stream, allow_redirects)

    web = Forbidding(latency=0.0)
    reader = PageReader(web, min_interval=0.1)
    results = []
    threads = [threading.Thread(target=lambda u=u: results.append(reader.read(u)))
               for u in [f"https://closed.example/{n}" for n in range(5)]]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert web.starts == []
    assert {r.status for r in results} == {"robots"}


# Hacker News content: up to twelve sites in flight, each site one story at a time


class TrackingReader:
    def __init__(self, latency):
        self.latency = latency
        self.active = {}
        self.most_at_once = 0
        self.same_site_overlap = False
        self.lock = threading.Lock()

    def read(self, url):
        host = url.split("/")[2]
        with self.lock:
            if self.active.get(host):
                self.same_site_overlap = True
            self.active[host] = self.active.get(host, 0) + 1
            self.most_at_once = max(self.most_at_once, sum(1 for n in self.active.values() if n))
        time.sleep(self.latency)
        with self.lock:
            self.active[host] -= 1
        return PageResult("ok", "Page text " * 30)


def test_hacker_news_pages_are_read_twelve_sites_at_a_time(conn):
    from techpulse import store

    now = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
    urls = [f"https://site{n}.example/post" for n in range(15)] + [f"https://busy.example/{n}" for n in range(3)]
    store.upsert(conn, [Article(
        source="hn", source_id=str(i), title=f"Story {i}", content=None, url=url, author="pg",
        categories=[], published_at=now - timedelta(hours=3), score=1,
    ) for i, url in enumerate(urls, 1)])
    items = {i: {"id": i, "type": "story", "url": url} for i, url in enumerate(urls, 1)}

    class Api:
        def get(self, url, params=None, timeout=None):
            return FakeResponse(json.dumps(items.get(int(url.rsplit("/", 1)[1].removesuffix(".json")))))

    reader = TrackingReader(latency=0.15)
    stats = enrich.poll(Fetcher(Api(), sleep=lambda _: None), reader, conn, base_url="https://hn.test/v0",
                        now=now, workers=12)
    assert stats["stories"] == 18 and stats["with_content"] == 18
    assert reader.most_at_once == 12
    assert not reader.same_site_overlap
    stored = conn.execute("SELECT count(*) FROM articles WHERE source = 'hn' AND content IS NOT NULL").fetchone()[0]
    assert stored == 18


# Dev.to: reaching back 30 days once, without a second collector


def test_a_deep_poll_reads_past_pages_with_nothing_new_down_to_the_cutoff(conn):
    from test_devto import Api, detail, fetcher_for, listed, NOW

    pages = {"python": [[listed(1)], [listed(2, hours_ago=24 * 5)], [listed(3, hours_ago=24 * 8)], [listed(4)]]}
    api = Api(pages, {i: detail(i) for i in range(1, 5)})
    fetcher, session, _ = fetcher_for(api)
    devto.poll(fetcher, conn, now=NOW, tags=["python"], page_size=1)  # takes 1 and 2, stops at 7 days
    conn.execute("DELETE FROM articles WHERE source_id = '2'")
    session.calls.clear()

    devto.poll(fetcher, conn, now=NOW, tags=["python"], page_size=1)
    assert [p["page"] for u, p in session.calls if u.endswith("/articles")] == [1]  # nothing new on page 1: stop

    session.calls.clear()
    devto.poll(fetcher, conn, now=NOW, tags=["python"], page_size=1, deep=True)
    assert [p["page"] for u, p in session.calls if u.endswith("/articles")] == [1, 2, 3]
    stored = {r[0] for r in conn.execute("SELECT source_id FROM articles WHERE source = 'devto'").fetchall()}
    assert stored == {"1", "2"}



# When a provider says "too many requests", slow down and stay slower


def test_a_429_halves_the_rate_of_the_shared_limiter():
    from fakes import FakeClock, FakeSession

    clock = FakeClock()
    replies = [FakeResponse("", status_code=429), FakeResponse("{}"), FakeResponse("{}"), FakeResponse("{}")]
    session = FakeSession(lambda url, params: replies.pop(0), clock)
    limiter = RateLimiter(0.2, clock=clock.monotonic, sleep=clock.sleep)
    fetcher = Fetcher(session, limiter=limiter, sleep=clock.sleep, base_delay=2.0)
    fetcher.get("https://dev.to/api/articles/1")
    assert limiter.min_interval == 0.4
    fetcher.get("https://dev.to/api/articles/2")
    fetcher.get("https://dev.to/api/articles/3")
    assert gaps(session.call_times)[-1] >= 0.4 - 1e-9


def test_repeated_429s_slow_down_to_a_floor_not_to_a_stop():
    from fakes import FakeClock

    clock = FakeClock()
    limiter = RateLimiter(0.2, clock=clock.monotonic, sleep=clock.sleep)
    for _ in range(10):  # a minute apart, so each one counts
        limiter.slow_down()
        clock.now += RateLimiter.SLOWDOWN_COOLDOWN
    assert limiter.min_interval == RateLimiter.SLOWEST_INTERVAL == 10.0


def test_other_failures_do_not_slow_the_limiter():
    from fakes import FakeClock, FakeSession

    clock = FakeClock()
    replies = [FakeResponse("", status_code=503), FakeResponse("{}")]
    session = FakeSession(lambda url, params: replies.pop(0), clock)
    limiter = RateLimiter(0.2, clock=clock.monotonic, sleep=clock.sleep)
    Fetcher(session, limiter=limiter, sleep=clock.sleep).get("https://dev.to/api/articles/1")
    assert limiter.min_interval == 0.2


def test_retry_after_given_as_a_date_is_honoured():
    from email.utils import format_datetime

    from fakes import FakeClock, FakeSession

    clock = FakeClock()
    when = format_datetime(datetime.now(timezone.utc) + timedelta(seconds=30), usegmt=True)
    replies = [FakeResponse("", status_code=429, headers={"Retry-After": when}), FakeResponse("{}")]
    session = FakeSession(lambda url, params: replies.pop(0), clock)
    Fetcher(session, sleep=clock.sleep, base_delay=2.0).get("https://dev.to/api/articles/1")
    assert 25 <= max(clock.sleeps) <= 31


def test_a_burst_of_429s_from_requests_in_flight_slows_down_once():
    from fakes import FakeClock

    clock = FakeClock()
    limiter = RateLimiter(0.2, clock=clock.monotonic, sleep=clock.sleep)
    for _ in range(8):  # eight requests that were already on their way all come back 429
        limiter.slow_down()
    assert limiter.min_interval == 0.4
    clock.now += RateLimiter.SLOWDOWN_COOLDOWN
    limiter.slow_down()
    assert limiter.min_interval == 0.8


def test_dev_to_reads_bodies_one_at_a_time():
    # Eight at once drew 429s from Dev.to within minutes (11/2026); one at a time ran for hours without one.
    assert devto.DETAIL_WORKERS == 1
