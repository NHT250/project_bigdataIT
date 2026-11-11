"""Plain text from the small markup the Hacker News API puts in post and comment text."""
import re
from html.parser import HTMLParser

BLOCKS = {"p", "br", "div", "li", "pre"}
HIDDEN = {"script", "style"}


class _TextCollector(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.hidden = 0
        self.preformatted = 0

    def handle_starttag(self, tag, attrs):
        if tag in HIDDEN:
            self.hidden += 1
        elif tag == "pre":
            self.preformatted += 1
            self.parts.append("\n\n")
        elif tag in BLOCKS:
            self.parts.append("\n\n")

    def handle_endtag(self, tag):
        if tag in HIDDEN:
            self.hidden = max(0, self.hidden - 1)
        elif tag == "pre":
            self.preformatted = max(0, self.preformatted - 1)
            self.parts.append("\n\n")

    def handle_data(self, data):
        if self.hidden:
            return
        if self.preformatted:
            self.parts.append(data.strip("\n"))
        else:
            self.parts.append(re.sub(r"\s+", " ", data))


def html_to_text(fragment):
    if not fragment:
        return ""
    collector = _TextCollector()
    collector.feed(fragment)
    collector.close()
    paragraphs = re.split(r"\n{2,}", "".join(collector.parts))
    cleaned = []
    for paragraph in paragraphs:
        lines = [line.rstrip() for line in paragraph.split("\n")]
        text = "\n".join(lines).strip("\n")
        if "\n" not in text:
            text = re.sub(r" {2,}", " ", text).strip()
        if text.strip():
            cleaned.append(text)
    return "\n\n".join(cleaned)
