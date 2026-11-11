"""Chunks with embeddings and model summaries, kept apart from the collector's table."""
from psycopg.rows import dict_row

EMBEDDING_DIMENSIONS = 384
# Splits a letter from a number joined to it ("QWen2.5" -> "QWen 2.5"). Documents and questions
# both go through it, so either spelling finds the other. Immutable, so a generated column can use it.
SPLIT_LETTERS_FROM_NUMBERS = r"regexp_replace({}, '([[:alpha:]])([[:digit:]])', '\1 \2', 'g')"
CURRENT_MD5 = "md5(a.title || E'\\n' || coalesce(a.content, ''))"

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS article_chunks (
    source      TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    chunk_index INT NOT NULL,
    text        TEXT NOT NULL,
    embedding   vector({dimensions}) NOT NULL,
    content_md5 TEXT NOT NULL,
    PRIMARY KEY (source, source_id, chunk_index),
    FOREIGN KEY (source, source_id) REFERENCES articles (source, source_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS article_chunks_embedding
    ON article_chunks USING hnsw (embedding vector_cosine_ops);

-- The keyword side of hybrid retrieval: title weighted above body, kept up to date by
-- PostgreSQL itself whenever the collector writes an article. A letter joined to a number is
-- split off first, so "QWen2.5" and "Qwen 2.5" index alike. The body is capped so a huge page
-- can't exceed the 1 MB tsvector limit.
ALTER TABLE articles ADD COLUMN IF NOT EXISTS search_document tsvector GENERATED ALWAYS AS (
    setweight(to_tsvector('english'::regconfig, {split_title}), 'A')
    || setweight(to_tsvector('english'::regconfig, {split_content}), 'D')
) STORED;

CREATE INDEX IF NOT EXISTS articles_search_document ON articles USING gin (search_document);

CREATE TABLE IF NOT EXISTS article_summaries (
    source      TEXT NOT NULL,
    source_id   TEXT NOT NULL,
    summary     TEXT NOT NULL,
    model       TEXT NOT NULL,
    content_md5 TEXT NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (source, source_id),
    FOREIGN KEY (source, source_id) REFERENCES articles (source, source_id) ON DELETE CASCADE
);
"""


SCHEMA = SCHEMA.format(
    dimensions=EMBEDDING_DIMENSIONS,
    split_title=SPLIT_LETTERS_FROM_NUMBERS.format("title"),
    split_content=SPLIT_LETTERS_FROM_NUMBERS.format("left(coalesce(content, ''), 300000)"),
)


def _drop_outdated_search_column(conn):
    """A search column made before letters and numbers were split is rebuilt with the new rule."""
    row = conn.execute("""
        SELECT generation_expression FROM information_schema.columns
        WHERE table_schema = current_schema() AND table_name = 'articles' AND column_name = 'search_document'
    """).fetchone()
    if row and "regexp_replace" not in (row[0] or ""):
        conn.execute("ALTER TABLE articles DROP COLUMN search_document")


def init_schema(conn):
    _drop_outdated_search_column(conn)
    conn.execute(SCHEMA)


def _rows(conn, query, params):
    with conn.cursor(row_factory=dict_row) as cursor:
        return cursor.execute(query, params).fetchall()


def _vector_literal(values):
    return "[" + ",".join(str(float(v)) for v in values) + "]"


def articles_needing_embedding(conn, limit):
    return _rows(conn, f"""
        SELECT a.source, a.source_id, a.title, a.content, {CURRENT_MD5} AS content_md5
        FROM articles a
        WHERE NOT EXISTS (
            SELECT 1 FROM article_chunks c
            WHERE c.source = a.source AND c.source_id = a.source_id AND c.content_md5 = {CURRENT_MD5}
        )
        ORDER BY a.published_at DESC
        LIMIT %s
    """, (limit,))


def articles_needing_summary(conn, limit):
    return _rows(conn, f"""
        SELECT a.source, a.source_id, a.title, a.content, {CURRENT_MD5} AS content_md5
        FROM articles a
        LEFT JOIN article_summaries s ON s.source = a.source AND s.source_id = a.source_id
        WHERE a.source = 'arxiv' AND a.content IS NOT NULL
          AND (s.content_md5 IS NULL OR s.content_md5 <> {CURRENT_MD5})
        ORDER BY a.published_at DESC
        LIMIT %s
    """, (limit,))


def save_chunks(conn, article, chunks, vectors):
    """Replace every stored chunk of the article in one transaction."""
    key = (article["source"], article["source_id"])
    with conn.transaction():
        conn.execute("DELETE FROM article_chunks WHERE source = %s AND source_id = %s", key)
        for index, (text, values) in enumerate(zip(chunks, vectors)):
            conn.execute(
                """
                INSERT INTO article_chunks (source, source_id, chunk_index, text, embedding, content_md5)
                VALUES (%s, %s, %s, %s, %s::vector, %s)
                """,
                (*key, index, text, _vector_literal(values), article["content_md5"]),
            )


def save_summary(conn, article, summary, model):
    conn.execute(
        """
        INSERT INTO article_summaries (source, source_id, summary, model, content_md5)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (source, source_id) DO UPDATE SET
            summary = EXCLUDED.summary,
            model = EXCLUDED.model,
            content_md5 = EXCLUDED.content_md5,
            created_at = now()
        """,
        (article["source"], article["source_id"], summary, model, article["content_md5"]),
    )


def search(conn, vector, k):
    """Return the k chunks closest to the vector by cosine similarity."""
    literal = _vector_literal(vector)
    return _rows(conn, """
        SELECT c.source, c.source_id, c.text, a.title, a.url, s.summary,
               1 - (c.embedding <=> %s::vector) AS score
        FROM article_chunks c
        JOIN articles a ON a.source = c.source AND a.source_id = c.source_id
        LEFT JOIN article_summaries s ON s.source = c.source AND s.source_id = c.source_id
        ORDER BY c.embedding <=> %s::vector
        LIMIT %s
    """, (literal, literal, k))


def indexed_article_count(conn):
    return conn.execute(
        "SELECT count(*) FROM (SELECT DISTINCT source, source_id FROM article_chunks) indexed"
    ).fetchone()[0]
