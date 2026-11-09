"""HTTP plumbing: honest User-Agent, rate limiting and exponential backoff."""
import time

import requests

RETRYABLE_STATUS = {429, 500, 502, 503, 504}


class FetchError(Exception):
    """A request failed for good: retries are exhausted or the error is permanent."""


def build_user_agent(email):
    if not email or not email.strip():
        raise ValueError("A contact email is required for the User-Agent")
    return f"TechPulse/1.0 (Van Lang University course project; {email.strip()})"


def make_session(email):
    session = requests.Session()
    session.headers["User-Agent"] = build_user_agent(email)
    return session


class RateLimiter:
    """Guarantees at least `min_interval` seconds between acquisitions."""

    def __init__(self, min_interval, clock=time.monotonic, sleep=time.sleep):
        self.min_interval = min_interval
        self.clock = clock
        self.sleep = sleep
        self.last = None

    def acquire(self):
        if self.last is not None:
            remaining = self.min_interval - (self.clock() - self.last)
            while remaining > 0:
                self.sleep(remaining)
                remaining = self.min_interval - (self.clock() - self.last)
        self.last = self.clock()


def _retry_after_seconds(response):
    value = response.headers.get("Retry-After", "")
    return float(value) if value.isdigit() else 0.0


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
            if attempt + 1 < attempts:
                backoff = min(self.base_delay * 2**attempt, self.max_delay)
                self.sleep(max(backoff, retry_after))
        raise FetchError(f"GET {url}: gave up after {attempts} attempts: {reason}")
