"""Dev.to in the feed: its own filter, its count, and the author's description."""
from techpulse import store as article_store
from techpulse.store import Article


def ids(response):
    return sorted(article["source_id"] for article in response.json()["articles"])


def add_post(conn, source_id, description="What the author says the post is about", hours_ago=1):
    from datetime import datetime, timedelta, timezone

    article_store.upsert(conn, [Article(
        source="devto", source_id=source_id, title=f"Post {source_id}", content="# Body\n\nText.",
        url=f"https://dev.to/u/{source_id}", author="Grace", categories=["python"],
        published_at=datetime.now(timezone.utc) - timedelta(hours=hours_ago), score=3,
        extra={"description": description, "comments": 1},
    )])


def test_feed_filters_to_dev_to(client, add_article, conn):
    add_article("paper", source="arxiv")
    add_article("story", source="hn")
    add_post(conn, "post")
    assert ids(client.get("/api/feed?source=devto")) == ["post"]


def test_feed_carries_the_authors_description_for_dev_to(client, conn):
    add_post(conn, "post", description="A hands-on guide")
    article = client.get("/api/feed?source=devto").json()["articles"][0]
    assert article["description"] == "A hands-on guide"
    assert article["summary"] is None


def test_dev_to_posts_count_as_having_content(client, conn):
    add_post(conn, "post")
    assert ids(client.get("/api/feed?source=content")) == ["post"]


def test_stats_count_dev_to(client, add_article, conn):
    add_article("paper", source="arxiv")
    add_post(conn, "post1")
    add_post(conn, "post2")
    stats = client.get("/api/stats").json()
    assert stats["counts"]["devto"] == 2
    assert stats["counts"]["content"] == 3
    assert stats["days"][-1]["devto"] == 2
