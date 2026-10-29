"""HTML → düz metin. Paragraf yapısı korunur (blok etiketleri boş satır olur) ki chunker paragraflardan bölebilsin."""
from __future__ import annotations

import re

_HTML_HINT = re.compile(r"<\s*(p|div|br|span|h[1-6]|li|ul|ol|table|tr|td|a|strong|b|em|i|section|article)\b", re.I)
_BLOCK_TAGS = (
    "p", "div", "section", "article", "header", "footer", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "ul", "ol", "li", "table", "tr",
)


def looks_like_html(text: str) -> bool:
    return bool(_HTML_HINT.search(text or ""))


def html_to_text(html: str) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html or "", "html.parser")
    for tag in soup(["script", "style", "noscript", "template"]):
        tag.decompose()
    # Kaynaktaki satır sonları/girintiler HTML'de anlamsızdır: boşluğa indir (<pre> hariç)
    for node in soup.find_all(string=True):
        if node.find_parent("pre") is None:
            node.replace_with(re.sub(r"\s+", " ", str(node)))
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for tag in soup.find_all(_BLOCK_TAGS):
        if tag.name == "li":
            # maddeler aynı paragrafta kalsın; listenin kendisi paragraf sınırıdır
            tag.insert_before("\n- ")
            continue
        tag.insert_before("\n\n")
        tag.insert_after("\n\n")
    for cell in soup.find_all(["td", "th"]):
        cell.insert_after(" | ")

    text = soup.get_text()
    lines = [re.sub(r"[ \t ]+", " ", line).strip() for line in text.splitlines()]
    text = "\n".join(lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def clean_text(value: str, content_format: str = "auto") -> str:
    """content_format: auto (HTML ise temizle) | html | plain."""
    value = value or ""
    if content_format == "html" or (content_format == "auto" and looks_like_html(value)):
        return html_to_text(value)
    return value.strip()
