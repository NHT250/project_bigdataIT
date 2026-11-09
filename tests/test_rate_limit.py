import requests

from fakes import FakeClock, FakeResponse, FakeSession, arxiv_entry, arxiv_feed
from techpulse import arxiv
from techpulse.net import RateLimiter


def gaps(times):
    return [later - earlier for earlier, later in zip(times, times[1:])]


def ok(url, params):
    return FakeResponse(arxiv_feed(arxiv_entry()))


def test_limiter_spaces_acquisitions_by_the_interval():
    clock = FakeClock()
    limiter = RateLimiter(3.0, clock=clock.monotonic, sleep=clock.sleep)
    times = []
    for _ in range(5):
        limiter.acquire()
        times.append(clock.now)
    assert all(gap >= 3.0 for gap in gaps(times))


def test_limiter_does_not_wait_when_interval_has_already_passed():
    clock = FakeClock()
    limiter = RateLimiter(3.0, clock=clock.monotonic, sleep=clock.sleep)
    limiter.acquire()
    clock.now += 10
    limiter.acquire()
    assert clock.sleeps == []


def test_consecutive_arxiv_requests_never_exceed_one_per_three_seconds():
    clock = FakeClock()
    session = FakeSession(ok, clock)
    fetcher = arxiv.make_fetcher(session, clock=clock.monotonic, sleep=clock.sleep)
    for start in range(0, 500, 100):
        arxiv.fetch_page(fetcher, start, 100)
    assert len(session.call_times) == 5
    assert all(gap >= 3.0 for gap in gaps(session.call_times))


def test_arxiv_retries_also_respect_the_rate_limit():
    clock = FakeClock()
    attempts = []

    def flaky(url, params):
        attempts.append(1)
        if len(attempts) <= 2:
            raise requests.ConnectionError("down")
        return ok(url, params)

    session = FakeSession(flaky, clock)
    fetcher = arxiv.make_fetcher(
        session, clock=clock.monotonic, sleep=clock.sleep, base_delay=0.01
    )
    arxiv.fetch_page(fetcher, 0, 100)
    assert len(session.call_times) == 3
    assert all(gap >= 3.0 for gap in gaps(session.call_times))
