"""Chunks, summaries and cosine search against a real PostgreSQL with pgvector."""
from rag import store


def vector(*head):
    """A 384-dimension vector whose first components are given."""
    return list(head) + [0.0] * (384 - len(head))


def pending_ids(conn):
    return [a["source_id"] for a in store.articles_needing_embedding(conn, 100)]


def test_new_article_needs_embedding(conn, add_article):
    add_article("1")
    assert pending_ids(conn) == ["1"]


def test_embedded_article_is_no_longer_pending(conn, add_article):
    add_article("1")
    article = store.articles_needing_embedding(conn, 1)[0]
    store.save_chunks(conn, article, ["text"], [vector(1.0)])
    assert pending_ids(conn) == []


def test_changed_content_is_embedded_again_and_replaces_old_chunks(conn, add_article):
    add_article("1", content="old")
    article = store.articles_needing_embedding(conn, 1)[0]
    store.save_chunks(conn, article, ["old a", "old b"], [vector(1.0), vector(0.0, 1.0)])

    add_article("1", content="new")
    article = store.articles_needing_embedding(conn, 1)[0]
    store.save_chunks(conn, article, ["new"], [vector(1.0)])

    assert [r["text"] for r in store.search(conn, vector(1.0), 5)] == ["new"]


def test_search_returns_the_nearest_chunks_first(conn, add_article):
    for source_id, v in [("far", vector(0.0, 1.0)), ("near", vector(1.0, 0.1)), ("exact", vector(1.0))]:
        add_article(source_id)
        article = [a for a in store.articles_needing_embedding(conn, 10) if a["source_id"] == source_id][0]
        store.save_chunks(conn, article, [source_id], [v])

    assert [r["source_id"] for r in store.search(conn, vector(1.0), 2)] == ["exact", "near"]


def test_search_includes_the_article_details_and_summary(conn, add_article):
    add_article("1", title="Paper", url="https://arxiv.org/abs/1")
    article = store.articles_needing_embedding(conn, 1)[0]
    store.save_chunks(conn, article, ["text"], [vector(1.0)])
    store.save_summary(conn, store.articles_needing_summary(conn, 1)[0], "One sentence.", "summarizer")

    row = store.search(conn, vector(1.0), 1)[0]

    assert (row["title"], row["url"], row["summary"]) == ("Paper", "https://arxiv.org/abs/1", "One sentence.")


def test_search_summary_is_empty_when_none_exists(conn, add_article):
    add_article("1", source="hn", content=None)
    article = store.articles_needing_embedding(conn, 1)[0]
    store.save_chunks(conn, article, ["text"], [vector(1.0)])
    assert store.search(conn, vector(1.0), 1)[0]["summary"] is None


def test_only_arxiv_articles_need_a_summary(conn, add_article):
    add_article("paper")
    add_article("story", source="hn", content=None)
    assert [a["source_id"] for a in store.articles_needing_summary(conn, 10)] == ["paper"]


def test_summarized_article_is_no_longer_pending(conn, add_article):
    add_article("1")
    store.save_summary(conn, store.articles_needing_summary(conn, 1)[0], "S", "summarizer")
    assert store.articles_needing_summary(conn, 10) == []


def test_changed_content_needs_a_new_summary(conn, add_article):
    add_article("1", content="old")
    store.save_summary(conn, store.articles_needing_summary(conn, 1)[0], "Old", "summarizer")
    add_article("1", content="new")
    store.save_summary(conn, store.articles_needing_summary(conn, 1)[0], "New", "summarizer")
    assert store.articles_needing_summary(conn, 10) == []


def test_indexed_article_count_counts_articles_not_chunks(conn, add_article):
    add_article("1")
    add_article("2")
    article = [a for a in store.articles_needing_embedding(conn, 10) if a["source_id"] == "1"][0]
    store.save_chunks(conn, article, ["a", "b"], [vector(1.0), vector(0.0, 1.0)])
    assert store.indexed_article_count(conn) == 1
