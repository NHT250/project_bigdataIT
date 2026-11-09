import json
import threading

import pytest
import requests

from fakes import FakeClock, FakeResponse, FakeSession
from techpulse import hn
from techpulse.net import Fetcher, FetchError
from techpulse.runner import run_forever

URL = "http://api.test/x"


def failing_then_ok(failures, error=None, response=None):
    remaining = [failures]

    def handler(url, params):
        if remaining[0] > 0:
            remaining[0] -= 1
            if error is not None:
                raise error
            return response
        return FakeResponse(b"ok")

    return handler


def fetcher_for(handler, clock, **kwargs):
    return Fetcher(FakeSession(handler), sleep=clock.sleep, **kwargs)


def test_network_error_is_retried_with_exponential_backoff():
    clock = FakeClock()
    fetcher = fetcher_for(
        failing_then_ok(3, error=requests.ConnectionError("down")), clock, base_delay=2.0
    )
    assert fetcher.get(URL).content == b"ok"
    assert clock.sleeps == [2.0, 4.0, 8.0]


def test_backoff_is_capped():
    clock = FakeClock()
    fetcher = fetcher_for(
        failing_then_ok(4, error=requests.Timeout("slow")),
        clock, base_delay=10.0, max_delay=25.0, max_attempts=5,
    )
    fetcher.get(URL)
    assert clock.sleeps == [10.0, 20.0, 25.0, 25.0]


@pytest.mark.parametrize("status", [429, 500, 503])
def test_transient_http_status_is_retried(status):
    clock = FakeClock()
    fetcher = fetcher_for(
        failing_then_ok(1, response=FakeResponse(status_code=status)), clock, base_delay=2.0
    )
    assert fetcher.get(URL).content == b"ok"


def test_retry_after_header_extends_the_wait():
    clock = FakeClock()
    busy = FakeResponse(status_code=503, headers={"Retry-After": "40"})
    fetcher = fetcher_for(failing_then_ok(1, response=busy), clock, base_delay=2.0)
    fetcher.get(URL)
    assert clock.sleeps == [40.0]


def test_client_error_is_not_retried():
    clock = FakeClock()
    session = FakeSession(lambda url, params: FakeResponse(status_code=404))
    with pytest.raises(FetchError):
        Fetcher(session, sleep=clock.sleep).get(URL)
    assert len(session.calls) == 1


def test_giving_up_raises_fetch_error_after_max_attempts():
    clock = FakeClock()

    def down(url, params):
        raise requests.ConnectionError("down")

    session = FakeSession(down)
    with pytest.raises(FetchError):
        Fetcher(session, sleep=clock.sleep, max_attempts=3).get(URL)
    assert len(session.calls) == 3


def run_loop(outcomes, interval=300):
    """Drive run_forever through scripted poll outcomes; return the waits it made."""
    stop = threading.Event()
    clock = FakeClock()
    script = iter(outcomes)
    polls = []

    def poll():
        polls.append(1)
        outcome = next(script)
        try:
            next(script_peek)
        except StopIteration:
            stop.set()
        if isinstance(outcome, Exception):
            raise outcome

    script_peek = iter(outcomes[1:])
    run_forever("test", poll, interval, stop, sleep=clock.sleep, clock=clock.monotonic)
    return polls, clock.sleeps


def test_job_survives_network_failures_and_keeps_polling():
    down = FetchError("network down")
    polls, _ = run_loop([down, down, None, down, None])
    assert len(polls) == 5


def test_job_survives_unexpected_errors_such_as_a_database_outage():
    polls, _ = run_loop([RuntimeError("db gone"), ValueError("bad row"), None])
    assert len(polls) == 3


def test_failed_polls_back_off_exponentially_then_recover():
    down = FetchError("network down")
    _, waits = run_loop([down, down, down, None, None])
    assert waits == [300, 600, 1200, 300]


def test_failure_backoff_is_capped():
    down = FetchError("network down")
    _, waits = run_loop([down] * 7)
    assert max(waits) == 1800


def test_poll_time_counts_towards_the_interval():
    stop = threading.Event()
    clock = FakeClock()
    polls = []

    def slow_poll():
        polls.append(1)
        clock.now += 100
        if len(polls) == 2:
            stop.set()

    run_forever("test", slow_poll, 300, stop, sleep=clock.sleep, clock=clock.monotonic)
    assert clock.sleeps == [200]


def hn_session(items):
    def handler(url, params):
        if url.endswith("topstories.json"):
            return FakeResponse(json.dumps(sorted(items)))
        if url.endswith("newstories.json"):
            return FakeResponse(b"[]")
        item_id = int(url.rsplit("/", 1)[1].removesuffix(".json"))
        if items[item_id] == "boom":
            raise requests.ConnectionError("down")
        return FakeResponse(json.dumps(items[item_id]))

    return FakeSession(handler)


def good_story(item_id):
    return {"id": item_id, "type": "story", "title": "t", "url": "http://x",
            "score": 1, "descendants": 0, "time": 1_800_000_000, "by": "a"}


def test_one_failing_hn_item_does_not_lose_the_others(conn):
    clock = FakeClock()
    session = hn_session({1: good_story(1), 2: "boom", 3: good_story(3)})
    stats = hn.poll(Fetcher(session, sleep=clock.sleep), conn)
    assert stats["stored"] == 2
    assert stats["failed"] == 1


def test_hn_poll_aborts_when_most_items_fail(conn):
    clock = FakeClock()
    session = hn_session({i: "boom" for i in range(1, 200)})
    with pytest.raises(FetchError):
        hn.poll(Fetcher(session, sleep=clock.sleep), conn)
