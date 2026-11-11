"""Groq's OpenAI-compatible chat API: request shape, streaming, retries on 429/503, failures."""
import json

import httpx
import pytest

from rag.answering import IncompleteReply
from rag.groq_client import GroqGenerator, ProviderUnavailable

URL = "https://api.groq.com/openai/v1/chat/completions"


def completion(text, finish_reason="stop"):
    return httpx.Response(200, json={
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": finish_reason}],
    })


def sse(*pieces, finish_reason="stop"):
    lines = [json.dumps({"choices": [{"index": 0, "delta": {"content": p}, "finish_reason": None}]}) for p in pieces]
    lines.append(json.dumps({"choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}]}))
    body = "".join(f"data: {line}\n\n" for line in lines) + "data: [DONE]\n\n"
    return httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode())


class Server:
    """Plays back one response (or exception) per request and records what was sent."""

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.requests = []

    def __call__(self, request):
        self.requests.append(request)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def groq(*outcomes, model="openai/gpt-oss-20b"):
    server = Server(*outcomes)
    waits = []
    client = httpx.Client(transport=httpx.MockTransport(server))
    return GroqGenerator(client, api_key="test-key", model=model, sleep=waits.append), server, waits


def sent(server, index=0):
    return json.loads(server.requests[index].content)


def busy(status=429, retry_after=None):
    headers = {"retry-after": str(retry_after)} if retry_after is not None else {}
    return httpx.Response(status, headers=headers, json={"error": {"message": "slow down"}})


# One reply at once


def test_reply_is_returned_from_one_request():
    generator, server, _ = groq(completion("It is a parser [1]."))
    assert generator("system rules", "the question") == "It is a parser [1]."
    request = server.requests[0]
    assert str(request.url) == URL
    assert request.headers["authorization"] == "Bearer test-key"
    body = sent(server)
    assert body["model"] == "openai/gpt-oss-20b"
    assert body["messages"] == [{"role": "system", "content": "system rules"},
                                {"role": "user", "content": "the question"}]
    assert body.get("stream") is not True


def test_gpt_oss_reasons_briefly_and_keeps_the_reasoning_out_of_the_reply():
    generator, server, _ = groq(completion("Hi [1]."))
    generator("s", "u")
    body = sent(server)
    assert body["reasoning_effort"] == "low"
    assert body["include_reasoning"] is False


def test_qwen_answers_without_reasoning_first():
    # Hidden reasoning still delays the first word: about 2.1 s against 1.0 s with it off (11/2026).
    generator, server, _ = groq(completion("Hi [1]."), model="qwen/qwen3.8-27b")
    generator("s", "u")
    assert sent(server)["reasoning_effort"] == "none"
    assert "reasoning_format" not in sent(server)


def test_a_cut_off_reply_counts_as_no_reply():
    generator, _, _ = groq(completion("It is a pa", finish_reason="length"))
    assert generator("s", "u") == ""


# Retries and failures


def test_rate_limit_is_retried_with_growing_waits():
    generator, server, waits = groq(busy(429), busy(503), completion("Hi [1]."))
    assert generator("s", "u") == "Hi [1]."
    assert len(server.requests) == 3
    assert len(waits) == 2 and waits[1] > waits[0]


def test_the_wait_groq_asks_for_is_honoured_but_capped():
    generator, _, waits = groq(busy(429, retry_after=2), busy(429, retry_after=600), completion("Hi [1]."))
    generator("s", "u")
    assert waits[0] == 2
    assert waits[1] <= 5


def test_three_retries_then_groq_counts_as_unavailable():
    generator, server, waits = groq(busy(503), busy(503), busy(429), busy(503))
    with pytest.raises(ProviderUnavailable):
        generator("s", "u")
    assert len(server.requests) == 4
    assert len(waits) == 3


@pytest.mark.parametrize("status", [400, 401, 404, 500])
def test_other_failures_are_not_retried(status):
    generator, server, _ = groq(httpx.Response(status, json={"error": {"message": "nope"}}))
    with pytest.raises(ProviderUnavailable, match=str(status)):
        generator("s", "u")
    assert len(server.requests) == 1


def test_network_failure_counts_as_unavailable():
    # A dropped connection is sent once more (see below), so the failure has to happen twice.
    generator, _, _ = groq(httpx.ConnectError("refused"), httpx.ConnectError("refused"))
    with pytest.raises(ProviderUnavailable):
        generator("s", "u")


def test_the_api_key_never_appears_in_an_error():
    generator, _, _ = groq(httpx.Response(401, json={"error": {"message": "Invalid API Key"}}))
    with pytest.raises(ProviderUnavailable) as caught:
        generator("s", "u")
    assert "test-key" not in str(caught.value)


# Streaming


def test_stream_yields_each_piece():
    generator, server, _ = groq(sse("It is ", "a parser ", "[1]."))
    assert list(generator.stream("s", "u")) == ["It is ", "a parser ", "[1]."]
    assert sent(server)["stream"] is True


def test_stream_retries_a_rate_limit_before_anything_was_sent():
    generator, _, waits = groq(busy(429), sse("Hi [1]."))
    assert list(generator.stream("s", "u")) == ["Hi [1]."]
    assert len(waits) == 1


def test_stream_cut_off_reports_an_incomplete_reply():
    generator, _, _ = groq(sse("It is a ", finish_reason="length"))
    with pytest.raises(IncompleteReply):
        list(generator.stream("s", "u"))


def test_stream_error_after_the_first_piece_counts_as_unavailable():
    body = ('data: {"choices":[{"index":0,"delta":{"content":"It is "},"finish_reason":null}]}\n\n'
            'data: {"error":{"message":"overloaded"}}\n\n')
    generator, _, _ = groq(httpx.Response(200, headers={"content-type": "text/event-stream"}, content=body.encode()))
    pieces = []
    with pytest.raises(ProviderUnavailable):
        for piece in generator.stream("s", "u"):
            pieces.append(piece)
    assert pieces == ["It is "]


# A kept-alive connection Groq has already closed


@pytest.mark.parametrize("dropped", [httpx.RemoteProtocolError("Server disconnected"), httpx.ConnectError("reset")])
def test_a_dropped_connection_is_tried_again_at_once_instead_of_giving_up(dropped):
    generator, server, waits = groq(dropped, completion("Hi [1]."))
    assert generator("s", "u") == "Hi [1]."
    assert len(server.requests) == 2
    assert waits == []


def test_a_stream_on_a_dropped_connection_is_tried_again_too():
    generator, _, _ = groq(httpx.RemoteProtocolError("Server disconnected"), sse("Hi [1]."))
    assert list(generator.stream("s", "u")) == ["Hi [1]."]


def test_a_connection_that_keeps_dropping_still_counts_as_unavailable():
    generator, server, _ = groq(httpx.ConnectError("reset"), httpx.ConnectError("reset"))
    with pytest.raises(ProviderUnavailable):
        generator("s", "u")
    assert len(server.requests) == 2
