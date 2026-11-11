"""The "With content" filter: arXiv papers always, Hacker News stories only with their own text."""
from techpulse import store as article_store


def ids(response):
    return sorted(article["source_id"] for article in response.json()["articles"])


def give_body(conn, source_id, content, page_status, comments=0):
    article_store.save_story_content(conn, source_id, content, page_status, comments)


def test_with_content_keeps_papers_and_stories_that_have_their_own_text(client, add_article, conn):
    add_article("paper", source="arxiv")
    add_article("page", source="hn", content=None)
    add_article("post", source="hn", content=None)
    add_article("comments-only", source="hn", content=None)
    add_article("blocked", source="hn", content=None)
    add_article("untouched", source="hn", content=None)
    give_body(conn, "page", "Linked page:\nThe article text.", "ok")
    give_body(conn, "post", "Post text:\nWhat is your setup?", "none")
    give_body(conn, "comments-only", "Top Hacker News comments:\n- Nice.", "robots", comments=1)
    give_body(conn, "blocked", None, "blocked")
    assert ids(client.get("/api/feed?source=content")) == ["page", "paper", "post"]


def test_with_content_is_not_the_default(client, add_article):
    add_article("untouched", source="hn", content=None)
    assert ids(client.get("/api/feed")) == ["untouched"]


def test_stats_count_each_filter(client, add_article, conn):
    add_article("paper1", source="arxiv")
    add_article("paper2", source="arxiv")
    add_article("page", source="hn", content=None)
    add_article("comments-only", source="hn", content=None)
    add_article("untouched", source="hn", content=None)
    give_body(conn, "page", "Linked page:\nText.", "ok")
    give_body(conn, "comments-only", "Top Hacker News comments:\n- Hm.", "no_text", comments=1)
    counts = client.get("/api/stats").json()["counts"]
    assert counts == {"all": 5, "arxiv": 2, "hn": 3, "devto": 0, "content": 3}
