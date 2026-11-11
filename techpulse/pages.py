"""Read the main text of a page a Hacker News story links to, politely.

The rules, in order: obey robots.txt (a site that can't serve one is left alone),
send the same honest User-Agent as every other request, wait between requests
to the same site (longer if its robots.txt asks), follow redirects one hop at a
time so each new site's robots.txt is checked too, and take no for an answer:
a refused, paywalled, oversized or non-HTML page is skipped, never worked around.
The text is only used for search and answers; the site shows titles and links.
"""
import threading
import time
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import trafilatura

from .net import TIMING_TOLERANCE

ROBOTS_NAME = "TechPulse"
MIN_INTERVAL_SECONDS = 5.0
MAX_CRAWL_DELAY_SECONDS = 60.0
MAX_BYTES = 2_000_000
MAX_CHARS = 20_000
MIN_CHARS = 200
MAX_REDIRECTS = 5
TIMEOUT_SECONDS = 15
HTML_TYPES = ("text/html", "application/xhtml+xml")
REFUSED = {401, 402, 403, 407, 429, 451}


@dataclass
class PageResult:
    # ok, robots, blocked, not_html, too_large, no_text, error or skipped
    status: str
    text: str = ""


class _Site:
    def __init__(self, robots, delay):
        self.robots = robots
        self.delay = delay
        self.last = None
        self.loaded = False
        # Held while waiting for a turn and while loading robots.txt, so two threads never
        # reach one site closer together than its delay.
        self.lock = threading.RLock()


class PageReader:
    def __init__(self, session, clock=time.monotonic, sleep=time.sleep, min_interval=MIN_INTERVAL_SECONDS,
                 max_bytes=MAX_BYTES, max_chars=MAX_CHARS):
        self.session = session
        self.clock = clock
        self.sleep = sleep
        self.min_interval = min_interval
        self.max_bytes = max_bytes
        self.max_chars = max_chars
        self.sites = {}
        self._sites_lock = threading.Lock()

    def read(self, url):
        for _ in range(MAX_REDIRECTS + 1):
            parts = urlsplit(url)
            if parts.scheme not in ("http", "https") or not parts.hostname:
                return PageResult("skipped")
            site = self._site(parts)
            if site.robots is None or not site.robots.can_fetch(ROBOTS_NAME, url):
                return PageResult("robots")
            self._wait(site)
            try:
                response = self.session.get(url, timeout=TIMEOUT_SECONDS, stream=True, allow_redirects=False)
            except Exception:
                return PageResult("error")
            try:
                if 300 <= response.status_code < 400 and response.headers.get("Location"):
                    url = urljoin(url, response.headers["Location"])
                    continue
                return self._page(response, url)
            finally:
                response.close()
        return PageResult("error")

    def _site(self, parts):
        origin = f"{parts.scheme}://{parts.netloc}"
        with self._sites_lock:
            site = self.sites.setdefault(origin, _Site(None, self.min_interval))
        with site.lock:
            if not site.loaded:
                self._load_robots(origin, site)
                site.loaded = True
        return site

    def _load_robots(self, origin, site):
        self._wait(site)
        try:
            response = self.session.get(f"{origin}/robots.txt", timeout=TIMEOUT_SECONDS)
        except Exception:
            return  # Can't tell what the site allows, so leave it alone.
        robots = RobotFileParser()
        if response.status_code in (401, 403):
            robots.disallow_all = True
        elif 400 <= response.status_code < 500:
            robots.allow_all = True  # No robots.txt means no restrictions.
        elif response.status_code >= 300:
            return
        else:
            robots.parse(response.text.splitlines())
        site.robots = robots
        crawl_delay = robots.crawl_delay(ROBOTS_NAME)
        if crawl_delay:
            site.delay = max(self.min_interval, min(float(crawl_delay), MAX_CRAWL_DELAY_SECONDS))

    def _wait(self, site):
        with site.lock:
            if site.last is not None:
                remaining = site.delay - (self.clock() - site.last)
                while remaining > TIMING_TOLERANCE:
                    self.sleep(remaining)
                    remaining = site.delay - (self.clock() - site.last)
            site.last = self.clock()

    def _page(self, response, url):
        if response.status_code in REFUSED:
            return PageResult("blocked")
        if response.status_code >= 300:
            return PageResult("error")
        content_type = response.headers.get("Content-Type", "").split(";")[0].strip().lower()
        if content_type not in HTML_TYPES:
            return PageResult("not_html")
        body = bytearray()
        for chunk in response.iter_content(chunk_size=65536):
            body.extend(chunk)
            if len(body) > self.max_bytes:
                return PageResult("too_large")
        html = bytes(body).decode(response.encoding or "utf-8", errors="replace")
        text = trafilatura.extract(html, url=url, favor_precision=True, include_comments=False,
                                   include_tables=False) or ""
        text = text.strip()
        if len(text) < MIN_CHARS:
            return PageResult("no_text")
        return PageResult("ok", text[:self.max_chars])
