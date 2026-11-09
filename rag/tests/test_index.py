"""Indexing jobs, with fake models and a real database."""
from rag import index, store


class FakeEmbedder:
    def __init__(self):
        self.texts = []
        self.tokenizer = self

    def __call__(self, text, add_special_tokens, return_offsets_mapping, truncation):
        import re
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}

    def encode(self, texts):
        self.texts.extend(texts)
        return [[1.0] + [0.0] * 383 for _ in texts]


def test_embed_pending_processes_every_article_across_batches(conn, add_article):
    for i in range(5):
        add_article(str(i))
    assert index.embed_pending(conn, FakeEmbedder(), batch_size=2) == 5
    assert store.indexed_article_count(conn) == 5


def test_embedded_text_starts_with_the_title(conn, add_article):
    add_article("1", title="The Title", content="The body")
    embedder = FakeEmbedder()
    index.embed_pending(conn, embedder)
    assert embedder.texts == ["The Title\n\nThe body"]


def test_hn_story_is_indexed_by_its_title(conn, add_article):
    add_article("1", source="hn", title="A story", content=None)
    embedder = FakeEmbedder()
    index.embed_pending(conn, embedder)
    assert embedder.texts == ["A story"]


def test_blank_article_still_finishes(conn, add_article):
    add_article("blank", title=" ", content=None)
    assert index.embed_pending(conn, FakeEmbedder()) == 1


def test_summarize_pending_processes_every_paper(conn, add_article):
    for i in range(3):
        add_article(str(i))
    assert index.summarize_pending(conn, lambda content: f"About {content}", batch_size=2) == 3
    assert store.articles_needing_summary(conn, 10) == []


def test_empty_summary_still_finishes(conn, add_article):
    add_article("1")
    assert index.summarize_pending(conn, lambda content: "") == 1
