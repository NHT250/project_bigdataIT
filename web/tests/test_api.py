"""What the website's pages and JSON endpoints promise."""
import pytest

from rag import answering

FOOTER = "Summarization model fine-tuned by us (Qwen 2.5 1.5B, QLoRA). Chat uses a hosted LLM with retrieval."


def ids(response, key="articles"):
    return [article["source_id"] for article in response.json()[key]]


@pytest.mark.parametrize("path,marker", [("/", "feed.js"), ("/chat", "chat.js"), ("/search", "search.js")])
def test_page_loads_with_its_script_and_the_footer(client, path, marker):
    response = client.get(path)
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert marker in response.text
    assert " ".join(FOOTER.split()) in " ".join(response.text.split())


@pytest.mark.parametrize("asset", ["app.css", "common.js", "feed.js", "chat.js", "search.js"])
def test_static_assets_are_served(client, asset):
    assert client.get(f"/static/{asset}").status_code == 200


def test_feed_returns_an_article_with_its_summary_and_link(client, add_article):
    add_article("1", title="Parsing", url="https://arxiv.org/abs/1", summary="We parse things.")
    article = client.get("/api/feed").json()["articles"][0]
    assert article["title"] == "Parsing"
    assert article["url"] == "https://arxiv.org/abs/1"
    assert article["summary"] == "We parse things."
    assert article["source"] == "arxiv"
    assert article["published_at"]


def test_feed_has_no_summary_for_an_unsummarized_article(client, add_article):
    add_article("1", source="hn")
    assert client.get("/api/feed").json()["articles"][0]["summary"] is None


def test_feed_links_an_hn_story_without_a_url_to_its_discussion(client, add_article):
    add_article("77", source="hn", url=None)
    assert client.get("/api/feed").json()["articles"][0]["url"] == "https://news.ycombinator.com/item?id=77"


def test_feed_shows_newest_first(client, add_article):
    add_article("old", hours_ago=5)
    add_article("new", hours_ago=1)
    assert ids(client.get("/api/feed?source=arxiv")) == ["new", "old"]


@pytest.mark.parametrize("source,expected", [("arxiv", ["a"]), ("hn", ["h"])])
def test_feed_filters_by_source(client, add_article, source, expected):
    add_article("a", source="arxiv")
    add_article("h", source="hn")
    assert ids(client.get(f"/api/feed?source={source}")) == expected


def test_all_feed_keeps_an_older_source_visible(client, add_article):
    for number in range(6):
        add_article(f"h{number}", source="hn", hours_ago=number + 1)
    add_article("a", source="arxiv", hours_ago=72)
    assert "a" in ids(client.get("/api/feed?limit=4"))


def test_feed_respects_the_limit(client, add_article):
    for number in range(3):
        add_article(str(number))
    assert len(ids(client.get("/api/feed?limit=2"))) == 2


@pytest.mark.parametrize("query", ["source=other", "limit=0", "limit=1000"])
def test_feed_rejects_bad_parameters(client, query):
    assert client.get(f"/api/feed?{query}").status_code == 422


def test_search_finds_a_word_in_the_title(client, add_article):
    add_article("1", title="Quantum compilers")
    add_article("2", title="Graph networks")
    assert ids(client.get("/api/search?q=quantum"), "results") == ["1"]


def test_search_finds_a_word_in_the_content(client, add_article):
    add_article("1", title="Plain", content="This paper studies photonics.")
    assert ids(client.get("/api/search?q=photonics"), "results") == ["1"]


def test_search_returns_the_summary_and_link(client, add_article):
    add_article("1", title="Quantum compilers", url="https://arxiv.org/abs/1", summary="We compile.")
    result = client.get("/api/search?q=quantum").json()["results"][0]
    assert (result["summary"], result["url"]) == ("We compile.", "https://arxiv.org/abs/1")


def test_search_ranks_a_title_match_above_a_content_match(client, add_article):
    add_article("body", title="Plain", content="We mention robotics once here.", hours_ago=1)
    add_article("title", title="Robotics today", content="Nothing else.", hours_ago=9)
    assert ids(client.get("/api/search?q=robotics"), "results") == ["title", "body"]


def test_search_with_no_match_is_empty(client, add_article):
    add_article("1", title="Graph networks")
    assert client.get("/api/search?q=zebra").json()["results"] == []


@pytest.mark.parametrize("query", ["", "%20%20"])
def test_blank_search_is_empty(client, add_article, query):
    add_article("1")
    assert client.get(f"/api/search?q={query}").json()["results"] == []


def test_search_treats_sql_as_plain_text(client, add_article):
    add_article("1", title="Graph networks")
    response = client.get("/api/search", params={"q": "'; DROP TABLE articles; --"})
    assert response.status_code == 200
    assert ids(client.get("/api/feed")) == ["1"]


def test_stats_count_articles_and_indexed_articles(client, add_article, conn):
    add_article("1", source="arxiv")
    add_article("2", source="hn")
    conn.execute(
        "INSERT INTO article_chunks VALUES ('arxiv', '1', 0, 't', %s::vector, 'x')",
        ("[" + ",".join(["0"] * 384) + "]",),
    )
    stats = client.get("/api/stats").json()
    assert (stats["total_articles"], stats["indexed_articles"]) == (2, 1)


def test_stats_give_fourteen_days_per_source(client, add_article):
    add_article("1", source="arxiv", hours_ago=0)
    add_article("2", source="hn", hours_ago=0)
    days = client.get("/api/stats").json()["days"]
    assert len(days) == 14
    assert (days[-1]["arxiv"], days[-1]["hn"]) == (1, 1)
    assert (days[0]["arxiv"], days[0]["hn"]) == (0, 0)


def test_chat_returns_the_answer_count_and_sources(client):
    body = client.post("/api/chat", json={"question": "What?"}).json()
    assert (body["status"], body["text"], body["searched_articles"]) == ("answered", "It works [1].", 7)
    assert body["sources"] == [{
        "number": 1, "title": "Paper", "url": "https://arxiv.org/abs/1", "summary": "A summary", "origin": "arxiv",
    }]


def test_chat_passes_the_trimmed_question(client, assistant):
    client.post("/api/chat", json={"question": "  What?  "})
    assert assistant.questions == ["What?"]


@pytest.mark.parametrize("question", ["", "   ", "x" * 1001])
def test_chat_rejects_an_unusable_question_without_calling_the_llm(client, assistant, question):
    assert client.post("/api/chat", json={"question": question}).status_code == 422
    assert assistant.questions == []


def test_chat_reports_a_missing_key(client, assistant):
    assistant.outcome = answering.MissingApiKey("GEMINI_API_KEY is not set.")
    response = client.post("/api/chat", json={"question": "What?"})
    assert (response.status_code, response.json()["detail"]) == (503, "GEMINI_API_KEY is not set.")


def test_chat_reports_an_llm_failure(client, assistant):
    assistant.outcome = answering.AnswerError("The Gemini API returned 503: busy")
    response = client.post("/api/chat", json={"question": "What?"})
    assert (response.status_code, response.json()["detail"]) == (502, "The Gemini API returned 503: busy")
