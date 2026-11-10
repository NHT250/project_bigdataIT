"""API tests run against a throwaway schema in the real PostgreSQL."""
import uuid
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

import psycopg
import pytest
from fastapi.testclient import TestClient

from rag import answering, settings, store
from techpulse import store as article_store
from techpulse.store import Article
from web.app import create_app


@pytest.fixture
def conn():
    schema = f"test_{uuid.uuid4().hex}"
    with psycopg.connect(settings.database_url(), autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        connection.execute(f'SET search_path TO "{schema}", public')
        article_store.init_schema(connection)
        store.init_schema(connection)
        yield connection
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def add_article(conn):
    def add(source_id, source="arxiv", title="A title", content="Some content",
            url="https://example.test/a", hours_ago=1, summary=None):
        published = datetime.now(timezone.utc) - timedelta(hours=hours_ago)
        article = Article(source=source, source_id=source_id, title=title, content=content, url=url,
                          author="Ada", categories=[], published_at=published)
        article_store.upsert(conn, [article])
        if summary:
            store.save_summary(conn, {"source": source, "source_id": source_id, "content_md5": "x"}, summary, "m")
    return add


class FakeAssistant:
    def __init__(self):
        self.questions = []
        self.histories = []
        self.outcome = answering.Answer("answered", "It works [1].", 7, [
            answering.Source(1, "Paper", "https://arxiv.org/abs/1", "A summary", "arxiv"),
        ])

    def __call__(self, question, history):
        self.questions.append(question)
        self.histories.append(history)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


@pytest.fixture
def assistant():
    return FakeAssistant()


@pytest.fixture
def client(conn, assistant):
    @contextmanager
    def connect():
        yield conn

    return TestClient(create_app(connect=connect, ask=assistant))
