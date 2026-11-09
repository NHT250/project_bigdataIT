"""Test doubles for the network and the clock."""
import json


class FakeClock:
    def __init__(self):
        self.now = 1000.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


class FakeResponse:
    def __init__(self, body=b"", status_code=200, headers=None):
        self.content = body if isinstance(body, bytes) else body.encode()
        self.status_code = status_code
        self.headers = headers or {}

    def json(self):
        return json.loads(self.content)


class FakeSession:
    """Calls handler(url, params) and records each call; handlers may raise."""

    def __init__(self, handler, clock=None):
        self.handler = handler
        self.clock = clock
        self.calls = []
        self.call_times = []

    def get(self, url, params=None, timeout=None):
        self.calls.append((url, params))
        if self.clock is not None:
            self.call_times.append(self.clock.now)
        return self.handler(url, params)


def arxiv_entry(arxiv_id="2411.00001v1", published="2026-11-02T10:00:00Z", title="A title"):
    return f"""
  <entry>
    <id>http://arxiv.org/abs/{arxiv_id}</id>
    <updated>2026-11-03T12:30:00Z</updated>
    <published>{published}</published>
    <title>{title}</title>
    <summary>An abstract.</summary>
    <author><name>Ada Lovelace</name></author>
    <link href="http://arxiv.org/abs/{arxiv_id}" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/{arxiv_id}" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>"""


def arxiv_feed(*entries):
    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom">'
        "<title>ArXiv Query</title>" + "".join(entries) + "</feed>"
    ).encode()
