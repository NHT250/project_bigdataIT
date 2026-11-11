"""The articles table: one row per (source, source_id)."""
from dataclasses import dataclass, field
from datetime import datetime

from psycopg.types.json import Jsonb

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    source       TEXT NOT NULL,
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
);

-- Replaced on every start, so a database made before a source was added accepts it.
ALTER TABLE articles DROP CONSTRAINT IF EXISTS articles_source_check;
ALTER TABLE articles ADD CONSTRAINT articles_source_check CHECK (source IN ('arxiv', 'hn', 'devto'));

-- One row per Hacker News story whose body has been looked for, found or not,
-- so a story is only read once.
CREATE TABLE IF NOT EXISTS hn_content (
    source_id   TEXT PRIMARY KEY,
    page_status TEXT NOT NULL,
    comments    INT NOT NULL,
    fetched_at  TIMESTAMPTZ NOT NULL DEFAULT now()
)
"""

UPSERT = """
INSERT INTO articles
    (source, source_id, title, content, url, author, categories, published_at, score, extra)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (source, source_id) DO UPDATE SET
    title = EXCLUDED.title,
    -- Hacker News polls carry no body; keep the one the enrichment job wrote.
    content = COALESCE(EXCLUDED.content, articles.content),
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


def stories_needing_content(conn, older_than, limit):
    """Hacker News stories never looked at, newest first."""
    return [row[0] for row in conn.execute("""
        SELECT a.source_id FROM articles a
        WHERE a.source = 'hn' AND a.published_at <= %s
          AND NOT EXISTS (SELECT 1 FROM hn_content c WHERE c.source_id = a.source_id)
        ORDER BY a.published_at DESC
        LIMIT %s
    """, (older_than, limit)).fetchall()]


def save_story_content(conn, source_id, content, page_status, comments):
    with conn.transaction():
        if content:
            conn.execute(
                "UPDATE articles SET content = %s WHERE source = 'hn' AND source_id = %s",
                (content, source_id),
            )
        conn.execute("""
            INSERT INTO hn_content (source_id, page_status, comments) VALUES (%s, %s, %s)
            ON CONFLICT (source_id) DO UPDATE SET
                page_status = EXCLUDED.page_status, comments = EXCLUDED.comments, fetched_at = now()
        """, (source_id, page_status, comments))


def article_urls(conn, source, source_ids):
    """{source_id: url} for the given articles of the source."""
    return dict(conn.execute(
        "SELECT source_id, url FROM articles WHERE source = %s AND source_id = ANY(%s)",
        (source, list(source_ids)),
    ).fetchall())


def existing_ids(conn, source, source_ids):
    """The ids among `source_ids` that are already stored for the source."""
    return {row[0] for row in conn.execute(
        "SELECT source_id FROM articles WHERE source = %s AND source_id = ANY(%s)",
        (source, list(source_ids)),
    ).fetchall()}


def count(conn, source):
    return conn.execute(
        "SELECT count(*) FROM articles WHERE source = %s", (source,)
    ).fetchone()[0]
