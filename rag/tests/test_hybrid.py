"""Hybrid retrieval: full-text (ts_rank_cd) and vector rankings fused with Reciprocal Rank Fusion."""
import pytest

from rag import ask, retrieval, store


def vector(*head):
    return list(head) + [0.0] * (384 - len(head))


# Reciprocal Rank Fusion on its own


def test_rrf_adds_up_the_scores_of_an_article_found_by_both_branches():
    fused = retrieval.rrf([["a", "b"], ["b", "c"]], k=60)
    scores = dict(fused)
    assert scores["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert scores["a"] == pytest.approx(1 / 61)
    assert [key for key, _ in fused][0] == "b"


def test_rrf_keeps_an_article_found_by_only_one_branch():
    fused = dict(retrieval.rrf([["a"], ["b", "c", "d"]], k=60))
    assert set(fused) == {"a", "b", "c", "d"}
    assert fused["d"] == pytest.approx(1 / 63)


def test_rrf_uses_ranks_not_the_branch_scores():
    # Two lists that agree on order fuse the same, whatever scale their scores had.
    assert [key for key, _ in retrieval.rrf([["x", "y", "z"], ["x", "y", "z"]])] == ["x", "y", "z"]


def test_rrf_breaks_ties_by_first_appearance():
    assert [key for key, _ in retrieval.rrf([["a", "b"], ["b", "a"]])] == ["a", "b"]


def test_rrf_default_k_is_60():
    assert retrieval.RRF_K == 60


# Against a real PostgreSQL with pgvector


@pytest.fixture
def archive(conn, add_article):
    """Three articles: the vector is near 'cats' and 'dogs', but only one names Qwen 2.5."""
    articles = [
        ("cats", "Why cats sleep all day", "Cats nap for most of the day to save energy.", vector(1.0)),
        ("dogs", "How dogs learn tricks", "Dogs learn tricks through repetition and treats.", vector(0.9, 0.1)),
        ("qwen", "Running a small model locally",
         "We ran Qwen 2.5 7B with 4-bit weights on a laptop GPU and measured tokens per second.", vector(0.0, 1.0)),
    ]
    for source_id, title, content, v in articles:
        add_article(source_id, title=title, content=content)
        article = [a for a in store.articles_needing_embedding(conn, 10) if a["source_id"] == source_id][0]
        store.save_chunks(conn, article, [f"{title}. {content}"], [v])
    return conn


def ids(rows):
    return [row["source_id"] for row in rows]


def test_keyword_branch_finds_a_proper_name_the_vector_misses(archive):
    assert ids(retrieval.retrieve(archive, "Qwen 2.5", vector(1.0), mode="bm25", k=5)) == ["qwen"]
    assert "qwen" not in ids(retrieval.retrieve(archive, "Qwen 2.5", vector(1.0), mode="dense", k=2))


def test_hybrid_keeps_the_proper_name_match_among_the_top_results(archive):
    found = ids(retrieval.retrieve(archive, "Qwen 2.5", vector(1.0), mode="hybrid", k=2))
    assert "qwen" in found


def test_dense_mode_ranks_by_cosine_similarity(archive):
    assert ids(retrieval.retrieve(archive, "anything", vector(1.0), mode="dense", k=3)) == ["cats", "dogs", "qwen"]


def test_an_article_found_by_both_branches_comes_first_in_hybrid(archive):
    # The vector favours cats a little; the words name dogs; dogs is high in both lists.
    found = ids(retrieval.retrieve(archive, "dogs learn tricks", vector(0.95, 0.05), mode="hybrid", k=3))
    assert found[0] == "dogs"


def test_each_article_appears_once_with_its_closest_chunk(conn, add_article):
    add_article("long", title="Long read", content="Part one. Part two.")
    article = store.articles_needing_embedding(conn, 1)[0]
    store.save_chunks(conn, article, ["far chunk", "near chunk"], [vector(0.0, 1.0), vector(1.0)])
    rows = retrieval.retrieve(conn, "long read", vector(1.0), mode="hybrid", k=5)
    assert ids(rows) == ["long"]
    assert rows[0]["text"] == "near chunk"
    assert {"source", "source_id", "text", "title", "url", "summary", "score"} <= set(rows[0])


def test_a_question_of_only_stopwords_still_returns_the_vector_results(archive):
    assert ids(retrieval.retrieve(archive, "what is the", vector(1.0), mode="hybrid", k=1)) == ["cats"]
    assert retrieval.retrieve(archive, "what is the", vector(1.0), mode="bm25", k=5) == []


def test_the_keyword_index_follows_a_changed_article(archive, add_article):
    add_article("cats", title="Why cats sleep all day", content="Now this article is about Mixtral instead.")
    assert ids(retrieval.retrieve(archive, "Mixtral", vector(0.0, 1.0), mode="bm25", k=5)) == ["cats"]


def add_indexed(conn, add_article, source_id, title, content):
    add_article(source_id, title=title, content=content)
    article = [a for a in store.articles_needing_embedding(conn, 10) if a["source_id"] == source_id][0]
    store.save_chunks(conn, article, [content], [vector(0.0, 0.0, 1.0)])


@pytest.mark.parametrize("written,asked", [
    ("We ran QWen2.5 7B on an RTX3050 laptop.", "Qwen 2.5 on an RTX 3050"),
    ("We ran Qwen 2.5 7B on an RTX 3050 laptop.", "Qwen2.5 on an RTX3050"),
])
def test_a_name_matches_whether_or_not_its_number_is_joined_on(conn, add_article, written, asked):
    add_indexed(conn, add_article, "model", "Benchmarks", written)
    add_indexed(conn, add_article, "other", "Cooking pasta", "Boil water and add salt.")
    assert ids(retrieval.retrieve(conn, asked, vector(1.0), mode="bm25", k=5)) == ["model"]


def test_a_search_column_from_before_the_split_is_rebuilt(conn, add_article):
    conn.execute("ALTER TABLE articles DROP COLUMN search_document")
    conn.execute("""ALTER TABLE articles ADD COLUMN search_document tsvector GENERATED ALWAYS AS (
        to_tsvector('english'::regconfig, title)) STORED""")
    store.init_schema(conn)
    add_indexed(conn, add_article, "model", "Benchmarks", "We ran QWen2.5 on a laptop.")
    assert ids(retrieval.retrieve(conn, "Qwen 2.5", vector(1.0), mode="bm25", k=5)) == ["model"]


def test_an_unknown_mode_is_a_clear_error(archive):
    with pytest.raises(ValueError, match="RETRIEVAL_MODE"):
        retrieval.retrieve(archive, "q", vector(1.0), mode="sparse", k=5)


# Choosing the mode


def test_retrieval_mode_defaults_to_hybrid():
    assert ask.retrieval_mode(lambda name, default="": default) == "hybrid"


@pytest.mark.parametrize("mode", ["hybrid", "dense", "bm25"])
def test_retrieval_mode_comes_from_the_environment(mode):
    assert ask.retrieval_mode(lambda name, default="": mode.upper() if name == "RETRIEVAL_MODE" else default) == mode


def test_an_unknown_retrieval_mode_is_refused_at_start():
    with pytest.raises(ValueError, match="RETRIEVAL_MODE"):
        ask.retrieval_mode(lambda name, default="": "sparse" if name == "RETRIEVAL_MODE" else default)
