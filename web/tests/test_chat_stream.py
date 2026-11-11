"""The streaming chat endpoint sends server-sent events: searching, pieces of the answer, then the result."""
import json
from contextlib import contextmanager

import pytest
from fastapi.testclient import TestClient

from rag import answering
from web.app import create_app

EARLIER = {"question": "What is new in graph neural networks?", "answer": "FiLSim [1].",
           "sources": [{"number": 1, "title": "FiLSim"}]}
FINAL = answering.Answer("answered", "It works [1].", 7,
                         [answering.Source(1, "Paper", "https://arxiv.org/abs/1", None, "arxiv")], "What?")


class FakeStreamingAssistant:
    def __init__(self):
        self.calls = []
        self.events = [
            {"type": "searching", "searched_for": "What?"},
            {"type": "delta", "text": "It works "},
            {"type": "delta", "text": "[1]."},
            {"type": "done", "answer": FINAL},
        ]
        self.fail_at_start = None
        self.fail_midway = None

    def __call__(self, question, history):
        self.calls.append((question, history))
        if self.fail_at_start:
            raise self.fail_at_start

        def play():
            for index, event in enumerate(self.events):
                if self.fail_midway and index == 2:
                    raise self.fail_midway
                yield event
        return play()


@pytest.fixture
def streaming():
    return FakeStreamingAssistant()


@pytest.fixture
def stream_client(conn, assistant, streaming):
    @contextmanager
    def connect():
        yield conn

    return TestClient(create_app(connect=connect, ask=assistant, ask_stream=streaming))


def read_events(response):
    return [json.loads(line[len("data: "):]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_stream_sends_each_event_in_order(stream_client):
    response = stream_client.post("/api/chat/stream", json={"question": "What?"})
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    events = read_events(response)
    assert [event["type"] for event in events] == ["searching", "delta", "delta", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "delta") == "It works [1]."


def test_stream_ends_with_the_checked_answer_and_its_sources(stream_client):
    done = read_events(stream_client.post("/api/chat/stream", json={"question": "What?"}))[-1]
    assert done["answer"]["status"] == "answered"
    assert done["answer"]["searched_articles"] == 7
    assert done["answer"]["sources"] == [
        {"number": 1, "title": "Paper", "url": "https://arxiv.org/abs/1", "summary": None, "origin": "arxiv"}
    ]


def test_stream_passes_the_question_and_earlier_turns(stream_client, streaming):
    stream_client.post("/api/chat/stream", json={"question": "  Tell me more about it  ", "history": [EARLIER]})
    question, history = streaming.calls[0]
    assert question == "Tell me more about it"
    assert history == [answering.Turn("What is new in graph neural networks?", "FiLSim [1].",
                                      [answering.CitedSource(1, "FiLSim")])]


def test_stream_rejects_an_unusable_question_without_calling_the_llm(stream_client, streaming):
    assert stream_client.post("/api/chat/stream", json={"question": "   "}).status_code == 422
    assert streaming.calls == []


def test_stream_reports_a_missing_key_before_streaming(stream_client, streaming):
    streaming.fail_at_start = answering.MissingApiKey("GEMINI_API_KEY is not set.")
    response = stream_client.post("/api/chat/stream", json={"question": "What?"})
    assert response.status_code == 503
    assert "GEMINI_API_KEY" in response.json()["detail"]


def test_stream_failure_midway_ends_with_a_readable_error_event(stream_client, streaming):
    streaming.fail_midway = answering.AnswerError("The Gemini API returned 500: boom")
    events = read_events(stream_client.post("/api/chat/stream", json={"question": "What?"}))
    assert [event["type"] for event in events] == ["searching", "delta", "error"]
    assert events[-1]["message"] == "The Gemini API returned 500: boom"


def test_stream_passes_on_the_follow_up_questions_after_the_answer(stream_client, streaming):
    streaming.events.append({"type": "suggestions", "questions": ["How large can FiLSim graphs be?"]})
    events = read_events(stream_client.post("/api/chat/stream", json={"question": "What?"}))
    assert [e["type"] for e in events][-2:] == ["done", "suggestions"]
    assert events[-1]["questions"] == ["How large can FiLSim graphs be?"]
