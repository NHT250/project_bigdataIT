from datetime import datetime, timezone

import pytest

from techpulse import arxiv

FEED = b"""<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <title type="html">ArXiv Query</title>
  <entry>
    <id>http://arxiv.org/abs/2411.01234v2</id>
    <updated>2026-11-03T12:30:00Z</updated>
    <published>2026-11-02T10:00:00Z</published>
    <title>Attention Is
      Still All You Need</title>
    <summary>  We study attention.
  It works &amp; scales.
    </summary>
    <author><name>Ada Lovelace</name></author>
    <author><name>Alan Turing</name><arxiv:affiliation xmlns:arxiv="http://arxiv.org/schemas/atom">Bletchley</arxiv:affiliation></author>
    <arxiv:comment xmlns:arxiv="http://arxiv.org/schemas/atom">10 pages</arxiv:comment>
    <link href="http://arxiv.org/abs/2411.01234v2" rel="alternate" type="text/html"/>
    <link title="pdf" href="http://arxiv.org/pdf/2411.01234v2" rel="related" type="application/pdf"/>
    <arxiv:primary_category xmlns:arxiv="http://arxiv.org/schemas/atom" term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.CL" scheme="http://arxiv.org/schemas/atom"/>
    <category term="cs.LG" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
  <entry>
    <id>http://arxiv.org/abs/cs/0112017v1</id>
    <updated>2001-12-20T00:00:00Z</updated>
    <published>2001-12-20T00:00:00Z</published>
    <title>Old style id</title>
    <summary>Old.</summary>
    <author><name>Grace Hopper</name></author>
    <category term="cs.AI" scheme="http://arxiv.org/schemas/atom"/>
  </entry>
</feed>
"""


def test_parses_every_entry():
    assert len(arxiv.parse_feed(FEED)) == 2


def test_source_id_drops_version_so_revisions_share_one_row():
    first = arxiv.parse_feed(FEED)[0]
    assert first.source == "arxiv"
    assert first.source_id == "2411.01234"


def test_old_style_id_keeps_archive_prefix():
    assert arxiv.parse_feed(FEED)[1].source_id == "cs/0112017"


def test_title_and_abstract_are_whitespace_normalised():
    first = arxiv.parse_feed(FEED)[0]
    assert first.title == "Attention Is Still All You Need"
    assert first.content == "We study attention. It works & scales."


def test_authors_and_categories_are_all_collected():
    first = arxiv.parse_feed(FEED)[0]
    assert first.author == "Ada Lovelace, Alan Turing"
    assert first.categories == ["cs.CL", "cs.LG"]


def test_timestamps_are_timezone_aware():
    first = arxiv.parse_feed(FEED)[0]
    assert first.published_at == datetime(2026, 11, 2, 10, 0, tzinfo=timezone.utc)
    assert first.extra["updated"] == "2026-11-03T12:30:00Z"


def test_links_are_extracted():
    first = arxiv.parse_feed(FEED)[0]
    assert first.url == "http://arxiv.org/abs/2411.01234v2"
    assert first.extra["pdf_url"] == "http://arxiv.org/pdf/2411.01234v2"


def test_score_is_empty_for_papers():
    assert arxiv.parse_feed(FEED)[0].score is None


def test_empty_feed_returns_no_articles():
    feed = b'<feed xmlns="http://www.w3.org/2005/Atom"><title>q</title></feed>'
    assert arxiv.parse_feed(feed) == []


def test_api_error_entry_is_raised_not_stored_as_a_paper():
    feed = b"""<feed xmlns="http://www.w3.org/2005/Atom">
      <entry><id>http://arxiv.org/api/errors#incorrect_id_format</id>
      <title>Error</title><summary>incorrect id format</summary></entry></feed>"""
    with pytest.raises(arxiv.ArxivApiError):
        arxiv.parse_feed(feed)
