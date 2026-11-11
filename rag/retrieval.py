"""Find the articles for a question: by meaning (pgvector), by words (full-text), or both fused.

Each branch ranks whole articles, up to 20. Hybrid fuses the two rankings with Reciprocal Rank
Fusion, score(d) = sum over branches of 1 / (60 + rank of d), which uses ranks only, because
cosine similarity and ts_rank_cd are on different scales. Whatever the mode, the passage shown
for an article is its chunk closest to the question's embedding.
"""
from rag.store import SPLIT_LETTERS_FROM_NUMBERS, _rows, _vector_literal

MODES = ("hybrid", "dense", "bm25")
RRF_K = 60
BRANCH_SIZE = 20
# HNSW returns at most ef_search rows, and several chunks can belong to one article.
CHUNKS_PER_ARTICLE = 5


def rrf(rankings, k=RRF_K):
    """Fuse ranked lists of keys into (key, score) pairs, best first; ties keep first appearance."""
    scores, first_seen = {}, {}
    for ranking in rankings:
        for rank, key in enumerate(ranking, 1):
            scores[key] = scores.get(key, 0.0) + 1 / (k + rank)
            first_seen.setdefault(key, len(first_seen))
    return sorted(scores.items(), key=lambda item: (-round(item[1], 12), first_seen[item[0]]))


def dense_ranking(conn, vector, size=BRANCH_SIZE):
    """Articles by their closest chunk's cosine distance."""
    literal = _vector_literal(vector)
    chunks = size * CHUNKS_PER_ARTICLE
    conn.execute(f"SET hnsw.ef_search = {max(chunks, 40)}")
    rows = _rows(conn, """
        SELECT source, source_id
        FROM (
            SELECT c.source, c.source_id, c.embedding <=> %s::vector AS distance
            FROM article_chunks c
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s
        ) nearest
        GROUP BY source, source_id
        ORDER BY min(distance)
        LIMIT %s
    """, (literal, literal, chunks, size))
    return [(row["source"], row["source_id"]) for row in rows]


def keyword_ranking(conn, text, size=BRANCH_SIZE):
    """Indexed articles by ts_rank_cd over title (weight A) and content (weight D).

    The question's words are joined with OR, so an article needn't contain every word; cover
    density ranks the ones holding more of them, closer together, higher.
    """
    split_question = SPLIT_LETTERS_FROM_NUMBERS.format("%s")
    rows = _rows(conn, f"""
        WITH q AS (
            SELECT nullif(replace(plainto_tsquery('english', {split_question})::text, ' & ', ' | '), '')::tsquery AS query
        )
        SELECT a.source, a.source_id
        FROM articles a, q
        WHERE q.query IS NOT NULL
          AND a.search_document @@ q.query
          AND EXISTS (SELECT 1 FROM article_chunks c WHERE c.source = a.source AND c.source_id = a.source_id)
        ORDER BY ts_rank_cd(a.search_document, q.query, 1) DESC, a.published_at DESC
        LIMIT %s
    """, (text, size))
    return [(row["source"], row["source_id"]) for row in rows]


def closest_chunks(conn, keys, vector):
    """For each article, its chunk closest to the vector, with the details the answer needs."""
    if not keys:
        return {}
    literal = _vector_literal(vector)
    rows = _rows(conn, """
        SELECT DISTINCT ON (c.source, c.source_id)
               c.source, c.source_id, c.text, a.title, a.url, s.summary,
               1 - (c.embedding <=> %s::vector) AS score
        FROM article_chunks c
        JOIN articles a ON a.source = c.source AND a.source_id = c.source_id
        LEFT JOIN article_summaries s ON s.source = c.source AND s.source_id = c.source_id
        WHERE (c.source, c.source_id) IN (SELECT * FROM unnest(%s::text[], %s::text[]))
        ORDER BY c.source, c.source_id, c.embedding <=> %s::vector
    """, (literal, [k[0] for k in keys], [k[1] for k in keys], literal))
    return {(row["source"], row["source_id"]): row for row in rows}


def ranked_articles(conn, text, vector, mode, k, branch_size=BRANCH_SIZE):
    """The top k (source, source_id) keys for the mode, best first."""
    if mode not in MODES:
        raise ValueError(f'RETRIEVAL_MODE must be one of {", ".join(MODES)}, not "{mode}".')
    rankings = []
    if mode in ("hybrid", "dense"):
        rankings.append(dense_ranking(conn, vector, branch_size))
    if mode in ("hybrid", "bm25"):
        rankings.append(keyword_ranking(conn, text, branch_size))
    return [key for key, _ in rrf(rankings)][:k]


def retrieve(conn, text, vector, mode="hybrid", k=5):
    """Rows like store.search returns, one per article, best first."""
    keys = ranked_articles(conn, text, vector, mode, k)
    chunks = closest_chunks(conn, keys, vector)
    return [chunks[key] for key in keys if key in chunks]
