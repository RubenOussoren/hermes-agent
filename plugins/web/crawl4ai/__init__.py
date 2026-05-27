"""Crawl4AI extract plugin — bundled, auto-loaded.

Crawl4AI is a free self-hosted companion for search-only providers such as
SearXNG. It exposes an HTTP API that turns rendered pages into clean Markdown
for ``web_extract`` calls.
"""

from __future__ import annotations

from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider


def register(ctx) -> None:
    """Register the Crawl4AI provider with the plugin context."""
    ctx.register_web_search_provider(Crawl4AIWebSearchProvider())
