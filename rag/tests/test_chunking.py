"""Sliding-window chunking: 500 tokens with a 50-token overlap."""
import re

from rag.chunking import chunk_text


class WordTokenizer:
    """One token per word, with the character offsets a real fast tokenizer gives."""

    def __call__(self, text, add_special_tokens, return_offsets_mapping, truncation):
        assert add_special_tokens is False and return_offsets_mapping is True and truncation is False
        return {"offset_mapping": [m.span() for m in re.finditer(r"\S+", text)]}


def words(count, start=0):
    return " ".join(f"Word{i}" for i in range(start, start + count))


def test_empty_text_has_no_chunks():
    assert chunk_text("   ", WordTokenizer()) == []


def test_text_within_the_limit_is_one_unchanged_chunk():
    text = words(500)
    assert chunk_text(text, WordTokenizer()) == [text]


def test_chunks_hold_at_most_500_tokens():
    chunks = chunk_text(words(1200), WordTokenizer())
    assert [len(c.split()) for c in chunks] == [500, 500, 300]


def test_consecutive_chunks_share_50_tokens():
    first, second = chunk_text(words(600), WordTokenizer())
    assert first.split()[-50:] == second.split()[:50]


def test_every_token_is_covered():
    chunks = chunk_text(words(1234), WordTokenizer())
    covered = {word for chunk in chunks for word in chunk.split()}
    assert covered == set(words(1234).split())


def test_no_trailing_chunk_of_only_overlap():
    chunks = chunk_text(words(950), WordTokenizer())
    assert len(chunks) == 2


def test_chunks_keep_the_original_text_between_tokens():
    text = "Line One,\n\n  Line Two"
    assert chunk_text(text, WordTokenizer()) == [text]
