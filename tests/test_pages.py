"""Reading linked pages politely: robots.txt, per-site spacing, size and type limits."""
from fakes import FakeClock
from techpulse.pages import PageReader

ARTICLE = """<!doctype html><html><head><title>Gemini 4 Carbon</title></head><body>
<nav>Home | Products | Blog</nav>
<article><h1>Gemini 4 Carbon is now available</h1>
<p>Today we are releasing Gemini 4 Carbon, our most efficient model for long documents and code.
It reads up to two million tokens and answers twice as fast as the previous generation.</p>
<p>Carbon was trained with a new mixture of experts design that activates fewer parameters per token,
which lowers the cost of every request for developers who build on the API.</p>
<p>The model is available today in the API and in the app, with a free tier for experimentation
and higher limits for paid plans, and it supports text, images and audio as input.</p>
</article><footer>Copyright Example Corp</footer></body></html>"""


class PageResponse:
    def __init__(self, body="", status_code=200, content_type="text/html; charset=utf-8", location=None):
        self.content = body.encode() if isinstance(body, str) else body
        self.status_code = status_code
        self.headers = {"Content-Type": content_type}
        if location:
            self.headers["Location"] = location
        self.encoding = "utf-8"
        self.closed = False

    @property
    def text(self):
        return self.content.decode()

    def iter_content(self, chunk_size=1):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def close(self):
        self.closed = True


class Site:
    """Serves fixed responses by URL and records every request with its time."""

    def __init__(self, pages, clock=None):
        self.pages = pages
        self.clock = clock
        self.requests = []

    def get(self, url, params=None, timeout=None, stream=False, allow_redirects=True):
        self.requests.append((url, self.clock.now if self.clock else None, allow_redirects))
        page = self.pages.get(url)
        if page is None:
            return PageResponse("not found", status_code=404)
        if isinstance(page, Exception):
            raise page
        return page


def reader(pages, **kwargs):
    clock = FakeClock()
    site = Site(pages, clock)
    return PageReader(site, clock=clock.monotonic, sleep=clock.sleep, **kwargs), site, clock


def fetched(site):
    return [url for url, _, _ in site.requests]


def test_article_text_is_extracted_without_navigation_or_footer():
    page_reader, _, _ = reader({"https://example.com/post": PageResponse(ARTICLE)})
    result = page_reader.read("https://example.com/post")
    assert result.status == "ok"
    assert "Gemini 4 Carbon" in result.text and "mixture of experts" in result.text
    assert "Copyright Example Corp" not in result.text


def test_robots_txt_is_read_before_the_page():
    page_reader, site, _ = reader({"https://example.com/post": PageResponse(ARTICLE)})
    page_reader.read("https://example.com/post")
    assert fetched(site) == ["https://example.com/robots.txt", "https://example.com/post"]


def test_page_disallowed_by_robots_txt_is_never_requested():
    robots = PageResponse("User-agent: *\nDisallow: /private/\n", content_type="text/plain")
    page_reader, site, _ = reader({
        "https://example.com/robots.txt": robots,
        "https://example.com/private/post": PageResponse(ARTICLE),
    })
    assert page_reader.read("https://example.com/private/post").status == "robots"
    assert fetched(site) == ["https://example.com/robots.txt"]


def test_robots_rules_for_techpulse_by_name_are_obeyed():
    robots = PageResponse("User-agent: TechPulse\nDisallow: /\n\nUser-agent: *\nAllow: /\n", content_type="text/plain")
    page_reader, site, _ = reader({"https://example.com/robots.txt": robots, "https://example.com/post": PageResponse(ARTICLE)})
    assert page_reader.read("https://example.com/post").status == "robots"
    assert "https://example.com/post" not in fetched(site)


def test_unreachable_robots_txt_means_the_site_is_left_alone():
    page_reader, site, _ = reader({
        "https://example.com/robots.txt": PageResponse("down", status_code=503),
        "https://example.com/post": PageResponse(ARTICLE),
    })
    assert page_reader.read("https://example.com/post").status == "robots"
    assert "https://example.com/post" not in fetched(site)


def test_robots_txt_is_read_once_per_site():
    page_reader, site, _ = reader({
        "https://example.com/a": PageResponse(ARTICLE),
        "https://example.com/b": PageResponse(ARTICLE),
    })
    page_reader.read("https://example.com/a")
    page_reader.read("https://example.com/b")
    assert fetched(site).count("https://example.com/robots.txt") == 1


def test_requests_to_one_site_are_spaced_out():
    page_reader, site, _ = reader({
        "https://example.com/a": PageResponse(ARTICLE),
        "https://example.com/b": PageResponse(ARTICLE),
    }, min_interval=5.0)
    page_reader.read("https://example.com/a")
    page_reader.read("https://example.com/b")
    times = [when for _, when, _ in site.requests]
    assert all(later - earlier >= 5.0 for earlier, later in zip(times, times[1:]))


def test_crawl_delay_in_robots_txt_is_honoured():
    robots = PageResponse("User-agent: *\nCrawl-delay: 12\n", content_type="text/plain")
    page_reader, site, _ = reader({
        "https://example.com/robots.txt": robots,
        "https://example.com/a": PageResponse(ARTICLE),
        "https://example.com/b": PageResponse(ARTICLE),
    }, min_interval=5.0)
    page_reader.read("https://example.com/a")
    page_reader.read("https://example.com/b")
    page_times = [when for url, when, _ in site.requests if url != "https://example.com/robots.txt"]
    assert page_times[1] - page_times[0] >= 12


def test_different_sites_do_not_wait_for_each_other():
    page_reader, site, clock = reader({
        "https://one.example/a": PageResponse(ARTICLE),
        "https://two.example/a": PageResponse(ARTICLE),
    }, min_interval=5.0)
    page_reader.read("https://one.example/a")
    finished_one = clock.now
    page_reader.read("https://two.example/a")
    first_request_to_two = next(when for url, when, _ in site.requests if url.startswith("https://two.example"))
    assert first_request_to_two == finished_one


def test_redirects_are_followed_one_hop_at_a_time_with_robots_checked_again():
    robots = PageResponse("User-agent: *\nDisallow: /\n", content_type="text/plain")
    page_reader, site, _ = reader({
        "https://short.example/x": PageResponse(status_code=301, location="https://blocked.example/post"),
        "https://blocked.example/robots.txt": robots,
        "https://blocked.example/post": PageResponse(ARTICLE),
    })
    assert page_reader.read("https://short.example/x").status == "robots"
    assert "https://blocked.example/post" not in fetched(site)
    assert all(allow is False for url, _, allow in site.requests if not url.endswith("robots.txt"))


def test_relative_redirect_is_resolved():
    page_reader, _, _ = reader({
        "https://example.com/old": PageResponse(status_code=302, location="/new"),
        "https://example.com/new": PageResponse(ARTICLE),
    })
    assert page_reader.read("https://example.com/old").status == "ok"


def test_too_many_redirects_give_up():
    pages = {f"https://example.com/{n}": PageResponse(status_code=302, location=f"/{n + 1}") for n in range(20)}
    page_reader, site, _ = reader(pages)
    assert page_reader.read("https://example.com/0").status == "error"
    assert len([u for u in fetched(site) if not u.endswith("robots.txt")]) <= 6


def test_pages_that_are_not_html_are_skipped():
    page_reader, _, _ = reader({"https://example.com/paper.pdf": PageResponse(b"%PDF-1.7", content_type="application/pdf")})
    assert page_reader.read("https://example.com/paper.pdf").status == "not_html"


def test_oversized_pages_are_dropped():
    page_reader, _, _ = reader({"https://example.com/huge": PageResponse("x" * 5000)}, max_bytes=1000)
    assert page_reader.read("https://example.com/huge").status == "too_large"


def test_refused_pages_are_not_retried_or_worked_around():
    page_reader, site, _ = reader({"https://example.com/paid": PageResponse("Subscribe", status_code=403)})
    assert page_reader.read("https://example.com/paid").status == "blocked"
    assert fetched(site).count("https://example.com/paid") == 1


def test_network_failure_is_an_error_not_a_crash():
    page_reader, _, _ = reader({"https://example.com/post": ConnectionError("reset")})
    assert page_reader.read("https://example.com/post").status == "error"


def test_page_without_article_text_is_reported_as_such():
    page_reader, _, _ = reader({"https://example.com/app": PageResponse("<html><body><div id=app></div></body></html>")})
    assert page_reader.read("https://example.com/app").status == "no_text"


def test_only_web_links_are_read():
    page_reader, site, _ = reader({})
    assert page_reader.read("ftp://example.com/file").status == "skipped"
    assert page_reader.read("javascript:alert(1)").status == "skipped"
    assert site.requests == []


def test_long_text_is_capped():
    long_article = ARTICLE.replace("</article>", "<p>" + "More detail about the model. " * 2000 + "</p></article>")
    page_reader, _, _ = reader({"https://example.com/post": PageResponse(long_article)}, max_chars=3000)
    assert len(page_reader.read("https://example.com/post").text) <= 3000
