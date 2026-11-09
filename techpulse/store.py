"""The articles table: one row per (source, source_id)."""
from dataclasses import dataclass, field
from datetime import datetime

from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    source       TEXT NOT NULL CHECK (source IN ('arxiv', 'hn')),
    source_id    TEXT NOT NULL,
    title        TEXT NOT NULL,
    content      TEXT,
    url          TEXT,
    author       TEXT,
    categories   TEXT[] NOT NULL DEFAULT '{}',
    published_at TIMESTAMPTZ NOT NULL,
    fetched_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    score        INT,
    extra        JSONB NOT NULL DEFAULT '{}',
    PRIMARY KEY (source, source_id)
)
"""

UPSERT = """
INSERT INTO articles
    (source, source_id, title, content, url, author, categories, published_at, score, extra)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (source, source_id) DO UPDATE SET
    title = EXCLUDED.title,
    content = EXCLUDED.content,
    url = EXCLUDED.url,
    author = EXCLUDED.author,
    categories = EXCLUDED.categories,
    published_at = EXCLUDED.published_at,
    fetched_at = now(),
    score = EXCLUDED.score,
    extra = EXCLUDED.extra
RETURNING (xmax = 0)
"""


@dataclass
class Article:
    source: str
    source_id: str
    title: str
    content: str | None
    url: str | None
    author: str | None
    categories: list[str]
    published_at: datetime
    score: int | None = None
    extra: dict = field(default_factory=dict)


def init_schema(conn):
    conn.execute(SCHEMA)


def upsert(conn, articles):
    """Insert or update the articles and return how many rows are new."""
    inserted = 0
    with conn.transaction():
        for a in articles:
            row = conn.execute(UPSERT, (
                a.source, a.source_id, a.title, a.content, a.url, a.author,
                a.categories, a.published_at, a.score, Jsonb(a.extra),
            )).fetchone()
            inserted += 1 if row[0] else 0
    return inserted


def count(conn, source):
    return conn.execute(
        "SELECT count(*) FROM articles WHERE source = %s", (source,)
    ).fetchone()[0]
