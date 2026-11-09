import http.server
import threading

import pytest

from techpulse import config
from techpulse.net import Fetcher, build_user_agent, make_session

EMAIL = "student@example.invalid"


@pytest.fixture
def wire():
    """A local HTTP server that records the headers actually sent."""
    seen = []

    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):
            seen.append(self.headers.get("User-Agent"))
            self.send_response(200)
            self.send_header("Content-Length", "2")
            self.end_headers()
            self.wfile.write(b"[]")

        def log_message(self, *args):
            pass

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/", seen
    server.shutdown()
    server.server_close()


def test_user_agent_matches_the_declared_format():
    assert build_user_agent(EMAIL) == (
        f"TechPulse/1.0 (Van Lang University course project; {EMAIL})"
    )


def test_user_agent_does_not_impersonate_a_browser():
    assert "Mozilla" not in build_user_agent(EMAIL)


def test_real_request_carries_the_declared_user_agent(wire):
    url, seen = wire
    Fetcher(make_session(EMAIL)).get(url)
    assert seen == [build_user_agent(EMAIL)]
    assert "Mozilla" not in seen[0]


@pytest.mark.parametrize("blank", ["", "   "])
def test_user_agent_refuses_a_blank_email(blank):
    with pytest.raises(ValueError):
        build_user_agent(blank)


def test_config_refuses_to_start_without_contact_email():
    with pytest.raises(SystemExit) as exc:
        config.load({"DATABASE_URL": "postgresql://x"})
    assert "CONTACT_EMAIL" in str(exc.value)


def test_config_reads_contact_email_from_env():
    loaded = config.load({"DATABASE_URL": "postgresql://x", "CONTACT_EMAIL": EMAIL})
    assert loaded.contact_email == EMAIL
