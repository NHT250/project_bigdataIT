"""HTTP plumbing: honest User-Agent, rate limiting and exponential backoff."""
import logging
import threading
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import requests

RETRYABLE_STATUS = {429, 500, 502, 503, 504}
TOO_MANY_REQUESTS = 429
# Waits stop within a nanosecond of their turn: floating-point subtraction can leave a remainder
# too small for a clock to ever advance past.
TIMING_TOLERANCE = 1e-9

log = logging.getLogger(__name__)


class FetchError(Exception):
    """A request failed for good: retries are exhausted or the error is permanent."""


PROJECT = "Van Lang University course project"


def build_user_agent(email, project=PROJECT):
    if not email or not email.strip():
        raise ValueError("A contact email is required for the User-Agent")
    return f"TechPulse/1.0 ({project}; {email.strip()})"


def make_session(email, project=PROJECT):
    session = requests.Session()
    session.headers["User-Agent"] = build_user_agent(email, project)
    return session


class RateLimiter:
    """Guarantees at least `min_interval` seconds between acquisitions, across threads too.

    A thread waits for its turn while holding the lock, so turns are handed out one by one.
    A provider that answers 429 has a lower limit than we assumed: slow_down() halves the rate
    for good (until the collector restarts), never slower than one request in 10 seconds. The
    requests already on their way when the first 429 arrives come back 429 too, so the rate is
    halved at most once a minute; it is never raised again, which would mean courting more 429s.
    """

    SLOWEST_INTERVAL = 10.0
    SLOWDOWN_COOLDOWN = 60.0

    def __init__(self, min_interval, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self.clock = clock
        self.sleep = sleep
        self.last = None
        self._slowed_at = None
        self._lock = threading.Lock()

    def acquire(self):
        with self._lock:
            self._wait_for_turn()

    def slow_down(self):
        with self._lock:
            now = self.clock()
            if self._slowed_at is not None and now - self._slowed_at < self.SLOWDOWN_COOLDOWN:
                return self.min_interval
            self._slowed_at = now
            self.min_interval = min(self.min_interval * 2, self.SLOWEST_INTERVAL)
            log.warning("Too many requests: now at most one every %.1fs", self.min_interval)
            return self.min_interval

    def _wait_for_turn(self):
        if self.last is not None:
            remaining = self.min_interval - (self.clock() - self.last)
            while remaining > TIMING_TOLERANCE:
                self.sleep(remaining)
                remaining = self.min_interval - (self.clock() - self.last)
        self.last = self.clock()


def _retry_after_seconds(response):
    """Retry-After as seconds, given either as a number or as an HTTP date."""
    value = response.headers.get("Retry-After", "").strip()
    if value.isdigit():
        return float(value)
    try:
        when = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return 0.0
    return max(0.0, (when - datetime.now(timezone.utc)).total_seconds())


class Fetcher:
    """GET with retries; every attempt, retries included, passes the rate limiter."""

    def __init__(self, session, limiter=None, sleep=time.sleep, max_attempts=5,
                 base_delay=2.0, max_delay=120.0, timeout=30):
        self.session = session
        self.limiter = limiter
        self.sleep = sleep
        self.max_attempts = max_attempts
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.timeout = timeout

    def get(self, url, params=None, max_attempts=None):
        attempts = max_attempts or self.max_attempts
        reason = "no attempt made"
        for attempt in range(attempts):
            if self.limiter is not None:
                self.limiter.acquire()
            retry_after = 0.0
            try:
                response = self.session.get(url, params=params, timeout=self.timeout)
            except requests.RequestException as exc:
                reason = repr(exc)
            else:
                if response.status_code < 400:
                    return response
                reason = f"HTTP {response.status_code}"
                if response.status_code not in RETRYABLE_STATUS:
                    raise FetchError(f"GET {url}: {reason}")
                retry_after = _retry_after_seconds(response)
                if response.status_code == TOO_MANY_REQUESTS and self.limiter is not None:
                    self.limiter.slow_down()
            if attempt + 1 < attempts:
                backoff = min(self.base_delay * 2**attempt, self.max_delay)
                self.sleep(max(backoff, retry_after))
        raise FetchError(f"GET {url}: gave up after {attempts} attempts: {reason}")
