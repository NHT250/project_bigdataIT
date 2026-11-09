"""The read queries behind the feed, search and statistics."""
from psycopg.rows import dict_row

from rag import answering

COLUMNS = "a.source, a.source_id, a.title, a.url, a.author, a.score, a.published_at, s.summary"
JOIN_SUMMARY = "LEFT JOIN article_summaries s ON s.source = a.source AND s.source_id = a.source_id"


def _rows(conn, query, params=None):
    with conn.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(query, params).fetchall()


def _articles(conn, query, params):
    return [{**row, "url": answering.source_link(row)} for row in _rows(conn, query, params)]


def feed(conn, source, limit):
    """Newest first. With every source, take turns so a slower source stays visible."""
    return _articles(conn, f"""
        SELECT {COLUMNS}
        FROM (
            SELECT *, row_number() OVER (PARTITION BY source ORDER BY published_at DESC) AS turn
            FROM articles
            WHERE %(source)s = 'all' OR source = %(source)s
        ) a
        {JOIN_SUMMARY}
        ORDER BY a.turn, a.published_at DESC
        LIMIT %(limit)s
    """, {"source": source, "limit": limit})


def search(conn, text, limit):
    """Full-text keyword search, with title matches ranked above body matches."""
    if not text.strip():
        return []
    return _articles(conn, f"""
        SELECT {COLUMNS}
        FROM articles a
        CROSS JOIN websearch_to_tsquery('english', %(text)s) query
        CROSS JOIN LATERAL (
            SELECT setweight(to_tsvector('english', a.title), 'A')
                || setweight(to_tsvector('english', coalesce(a.content, '')), 'D') AS document
        ) d
        {JOIN_SUMMARY}
        WHERE d.document @@ query
        ORDER BY ts_rank(d.document, query) DESC, a.published_at DESC
        LIMIT %(limit)s
    """, {"text": text, "limit": limit})


def stats(conn):
    totals = _rows(conn, """
        SELECT (SELECT count(*) FROM articles) AS total_articles,
               (SELECT count(*) FROM (SELECT DISTINCT source, source_id FROM article_chunks) i) AS indexed_articles
    """)[0]
    days = _rows(conn, """
        SELECT d::date AS day,
               count(*) FILTER (WHERE a.source = 'arxiv') AS arxiv,
               count(*) FILTER (WHERE a.source = 'hn') AS hn
        FROM generate_series(
            (now() AT TIME ZONE 'UTC')::date - 13, (now() AT TIME ZONE 'UTC')::date, interval '1 day'
        ) d
        LEFT JOIN articles a ON (a.published_at AT TIME ZONE 'UTC')::date = d::date
        GROUP BY d
        ORDER BY d
    """)
    return {**totals, "days": days}
