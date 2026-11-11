"""How often each source is polled: lightly, now the archive is complete, but never off."""
from techpulse import __main__ as collector
from techpulse import hn


def test_each_source_has_its_own_interval():
    assert collector.POLL_INTERVALS == {"arxiv": 3600, "hn": 1200, "devto": 900}


def test_no_source_is_switched_off():
    assert all(0 < seconds <= 3600 for seconds in collector.POLL_INTERVALS.values())
    assert 0 < collector.CONTENT_INTERVAL_SECONDS <= 3600


def test_hacker_news_reads_items_two_at_a_time():
    assert hn.WORKERS == 2
