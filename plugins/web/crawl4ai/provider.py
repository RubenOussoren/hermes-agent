"""Crawl4AI page extraction provider.

Search-only providers such as SearXNG are excellent for discovery but cannot
fetch arbitrary page content for ``web_extract``. Crawl4AI fills that gap with a
self-hosted, open-source extraction API that returns LLM-friendly Markdown.

Config keys this provider responds to::

    web:
      extract_backend: "crawl4ai"    # explicit per-capability
      backend: "crawl4ai"            # shared fallback

Env var::

    CRAWL4AI_URL=http://localhost:11235
    CRAWL4AI_API_TOKEN=<optional bearer token>
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional

from agent.web_search_provider import WebSearchProvider
from tools.website_policy import check_website_access

logger = logging.getLogger(__name__)


class Crawl4AIWebSearchProvider(WebSearchProvider):
    """Extract page content via a user-hosted Crawl4AI server."""

    @property
    def name(self) -> str:
        return "crawl4ai"

    @property
    def display_name(self) -> str:
        return "Crawl4AI"

    def is_available(self) -> bool:
        """Return True when ``CRAWL4AI_URL`` is set."""
        return bool(os.getenv("CRAWL4AI_URL", "").strip())

    def supports_search(self) -> bool:
        return False

    def supports_extract(self) -> bool:
        return True

    async def extract(self, urls: List[str], **kwargs: Any) -> List[Dict[str, Any]]:
        """Extract readable content from URLs using Crawl4AI's ``/md`` endpoint."""
        import asyncio
        import httpx

        format = kwargs.get("format")
        base_url = os.getenv("CRAWL4AI_URL", "").strip().rstrip("/")
        if not base_url:
            return [
                {
                    "url": url,
                    "title": "",
                    "content": "",
                    "raw_content": "",
                    "metadata": {},
                    "error": "CRAWL4AI_URL is not set",
                }
                for url in urls
            ]

        headers = {"Accept": "application/json"}
        api_token = os.getenv("CRAWL4AI_API_TOKEN", "").strip()
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        async with httpx.AsyncClient(
            timeout=httpx.Timeout(60.0, connect=10.0),
            headers=headers,
        ) as client:
            tasks = [self._extract_one(client, base_url, url, format=format) for url in urls]
            return list(await asyncio.gather(*tasks))

    async def _extract_one(
        self,
        client: Any,
        base_url: str,
        url: str,
        *,
        format: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Extract one URL and normalize Crawl4AI's response to Hermes shape."""
        import httpx

        policy_block = check_website_access(url)
        if policy_block:
            return {
                "url": url,
                "title": "",
                "content": "",
                "raw_content": "",
                "metadata": {"blocked_by": policy_block},
                "error": policy_block.get("message", "Blocked by website policy"),
            }

        filter_type = "raw" if (format or "").lower() == "html" else "fit"
        payload: Dict[str, Any] = {"url": url, "f": filter_type, "c": "0"}

        try:
            resp = await client.post(f"{base_url}/md", json=payload)
            resp.raise_for_status()
            data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.warning(
                "Crawl4AI HTTP error for %s: HTTP %s",
                url,
                exc.response.status_code,
            )
            return self._error_result(
                url,
                f"Crawl4AI returned HTTP {exc.response.status_code}",
            )
        except httpx.RequestError as exc:
            logger.warning(
                "Crawl4AI request error for %s: %s",
                url,
                type(exc).__name__,
            )
            return self._error_result(
                url,
                "Could not reach Crawl4AI at configured CRAWL4AI_URL",
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Crawl4AI response parse error for %s: %s", url, exc)
            return self._error_result(url, "Could not parse Crawl4AI response as JSON")

        if data.get("success") is False:
            error = data.get("error") or data.get("detail") or "Crawl4AI extraction failed"
            return self._error_result(url, str(error), metadata={"crawl4ai": data})

        markdown = self._coerce_markdown(data.get("markdown"))
        title = str(data.get("title") or self._title_from_markdown(markdown) or url)
        resolved_url = str(data.get("url") or url)
        metadata = {
            "source": "crawl4ai",
            "filter": data.get("filter"),
            "query": data.get("query"),
            "cache": data.get("cache"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return {
            "url": resolved_url,
            "title": title,
            "content": markdown,
            "raw_content": markdown,
            "metadata": metadata,
        }

    @staticmethod
    def _coerce_markdown(markdown: Any) -> str:
        """Return a Markdown string from Crawl4AI's possible payload variants."""
        if isinstance(markdown, str):
            return markdown
        if isinstance(markdown, dict):
            for key in ("fit_markdown", "raw_markdown", "markdown"):
                value = markdown.get(key)
                if isinstance(value, str) and value.strip():
                    return value
        return ""

    @staticmethod
    def _title_from_markdown(markdown: str) -> str:
        for line in (markdown or "").splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                return stripped.lstrip("#").strip()
        return ""

    @staticmethod
    def _error_result(
        url: str,
        error: str,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "url": url,
            "title": "",
            "content": "",
            "raw_content": "",
            "metadata": metadata or {},
            "error": error,
        }

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": "Crawl4AI",
            "badge": "free · self-hosted",
            "tag": "Open-source LLM-friendly page extraction. Point CRAWL4AI_URL at your instance.",
            "env_vars": [
                {
                    "key": "CRAWL4AI_URL",
                    "prompt": "Crawl4AI server URL (e.g. http://localhost:11235)",
                    "url": "https://github.com/unclecode/crawl4ai",
                },
                {
                    "key": "CRAWL4AI_API_TOKEN",
                    "prompt": "Crawl4AI bearer token for protected instances",
                    "url": "https://github.com/unclecode/crawl4ai",
                    "optional": True,
                },
            ],
        }
