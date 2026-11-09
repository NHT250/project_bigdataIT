"""DB tests run in a throwaway schema, never in the live tables."""
import uuid
from datetime import datetime, timezone

import psycopg
import pytest

from rag import settings, store
from techpulse import store as article_store
from techpulse.store import Article


@pytest.fixture
def conn():
    schema = f"test_{uuid.uuid4().hex}"
    with psycopg.connect(settings.database_url(), autocommit=True) as connection:
        connection.execute(f'CREATE SCHEMA "{schema}"')
        # public stays on the path for the vector type.
        connection.execute(f'SET search_path TO "{schema}", public')
        article_store.init_schema(connection)
        store.init_schema(connection)
        yield connection
        connection.execute(f'DROP SCHEMA "{schema}" CASCADE')


@pytest.fixture
def add_article(conn):
    def add(source_id, source="arxiv", title="A title", content="Some content", url="https://example.test/a"):
        article_store.upsert(conn, [Article(
            source=source, source_id=source_id, title=title, content=content, url=url,
            author=None, categories=[], published_at=datetime(2026, 11, 1, tzinfo=timezone.utc),
        )])
    return add
