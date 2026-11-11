"""Hacker News API text fields carry a little markup that has to become plain text."""
from techpulse.text import html_to_text


def test_paragraphs_become_blank_lines():
    assert html_to_text("First line.<p>Second line.") == "First line.\n\nSecond line."


def test_entities_are_decoded():
    assert html_to_text("Rust &amp; Go &#x27;fast&#x27; &gt; C") == "Rust & Go 'fast' > C"


def test_link_text_is_kept_and_markup_dropped():
    assert html_to_text('See <a href="https://example.com/x" rel="nofollow">the docs</a> and <i>this</i>.') == (
        "See the docs and this."
    )


def test_code_blocks_keep_their_lines():
    assert html_to_text("Try:<p><pre><code>  a = 1\n  b = 2\n</code></pre>") == "Try:\n\n  a = 1\n  b = 2"


def test_empty_or_missing_text_is_empty():
    assert html_to_text(None) == ""
    assert html_to_text("") == ""


def test_scripts_are_not_turned_into_text():
    assert html_to_text("Hi<script>alert(1)</script> there") == "Hi there"
