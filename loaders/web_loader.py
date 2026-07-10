"""Web sayfası loader: Trafilatura -> Playwright fallback."""
from __future__ import annotations

from typing import Callable

from core.types import Document
from core.utils import domain_of

import config

from .base import Loader


class WebLoader(Loader):
    source_type = "web"

    def load(
        self,
        source: str,
        whisper_language: str | None = None,
        check_cancelled: Callable[[], bool] | None = None,
    ) -> Document:
        if check_cancelled and check_cancelled():
            from core.types import OperationCancelled
            raise OperationCancelled("İşlem kullanıcı tarafından iptal edildi.")

        text, title, used = self._try_trafilatura(source)

        if not text or len(text) < 80:
            pw_text, pw_title = self._try_playwright(source)
            if pw_text and len(pw_text) > len(text or ""):
                text, title, used = pw_text, pw_title or title, "playwright"

        return Document(
            text=text or "",
            source_type=self.source_type,
            source_uri=source,
            title=title,
            language=None,
            extra={
                "extractor": used,
                "domain": domain_of(source),
            },
        )

    def _try_trafilatura(self, url: str) -> tuple[str, str | None, str]:
        try:
            import trafilatura
        except ImportError:
            return "", None, "none"

        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return "", None, "trafilatura-empty"

        text = trafilatura.extract(
            downloaded,
            include_comments=False,
            include_tables=True,
            favor_recall=True,
        ) or ""

        meta = trafilatura.extract_metadata(downloaded)
        title = getattr(meta, "title", None) if meta else None
        return text, title, "trafilatura"

    def _try_playwright(self, url: str) -> tuple[str, str | None]:
        try:
            from playwright.sync_api import sync_playwright
        except ImportError:
            return "", None

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(user_agent=config.WEB_USER_AGENT)
                page = context.new_page()
                page.set_default_timeout(config.PLAYWRIGHT_TIMEOUT_MS)
                page.goto(url, wait_until="domcontentloaded")
                page.wait_for_load_state("networkidle", timeout=config.PLAYWRIGHT_TIMEOUT_MS)
                html = page.content()
                title = page.title()
                browser.close()
        except Exception:
            return "", None

        try:
            import trafilatura
            text = trafilatura.extract(html, include_comments=False, favor_recall=True) or ""
        except Exception:
            from bs4 import BeautifulSoup
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            text = soup.get_text("\n", strip=True)

        return text, title
