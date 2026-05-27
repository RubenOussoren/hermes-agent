"""Tests for the Crawl4AI web extraction provider."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import patch

import httpx

_REAL_ASYNC_CLIENT = httpx.AsyncClient


def _mock_async_client(handler):
    transport = httpx.MockTransport(handler)
    return _REAL_ASYNC_CLIENT(transport=transport, timeout=httpx.Timeout(60.0, connect=10.0))


class TestCrawl4AIProviderAvailability:
    def test_available_when_url_set(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://localhost:11235")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        assert Crawl4AIWebSearchProvider().is_available() is True

    def test_not_available_when_url_missing(self, monkeypatch):
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        assert Crawl4AIWebSearchProvider().is_available() is False

    def test_capability_flags(self):
        from agent.web_search_provider import WebSearchProvider
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        provider = Crawl4AIWebSearchProvider()
        assert issubclass(Crawl4AIWebSearchProvider, WebSearchProvider)
        assert provider.name == "crawl4ai"
        assert provider.supports_search() is False
        assert provider.supports_extract() is True


class TestCrawl4AIProviderExtract:
    def test_extract_happy_path_returns_normalized_result(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://crawl4ai:11235/")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        requests = []

        def handler(request: httpx.Request) -> httpx.Response:
            requests.append(request)
            assert str(request.url) == "http://crawl4ai:11235/md"
            payload = json.loads(request.content.decode())
            assert payload == {"url": "https://example.com", "f": "fit", "c": "0"}
            return httpx.Response(
                200,
                json={
                    "url": "https://example.com",
                    "filter": "fit",
                    "markdown": "# Example Title\n\nReadable content",
                    "success": True,
                },
            )

        with patch("httpx.AsyncClient", side_effect=lambda **_: _mock_async_client(handler)):
            result = asyncio.run(Crawl4AIWebSearchProvider().extract(["https://example.com"]))

        assert len(requests) == 1
        assert result == [
            {
                "url": "https://example.com",
                "title": "Example Title",
                "content": "# Example Title\n\nReadable content",
                "raw_content": "# Example Title\n\nReadable content",
                "metadata": {"source": "crawl4ai", "filter": "fit"},
            }
        ]

    def test_extract_html_format_uses_raw_filter(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://crawl4ai:11235")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        seen_payloads = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen_payloads.append(json.loads(request.content.decode()))
            return httpx.Response(200, json={"markdown": "raw", "success": True})

        with patch("httpx.AsyncClient", side_effect=lambda **_: _mock_async_client(handler)):
            result = asyncio.run(
                Crawl4AIWebSearchProvider().extract(["https://example.com"], format="html")
            )

        assert seen_payloads[0]["f"] == "raw"
        assert result[0]["content"] == "raw"

    def test_missing_url_returns_per_url_error(self, monkeypatch):
        monkeypatch.delenv("CRAWL4AI_URL", raising=False)
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        result = asyncio.run(Crawl4AIWebSearchProvider().extract(["https://example.com"]))

        assert result[0]["url"] == "https://example.com"
        assert "CRAWL4AI_URL" in result[0]["error"]

    def test_http_error_returns_per_url_error(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://crawl4ai:11235")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(503, json={"detail": "not ready"})

        with patch("httpx.AsyncClient", side_effect=lambda **_: _mock_async_client(handler)):
            result = asyncio.run(Crawl4AIWebSearchProvider().extract(["https://example.com"]))

        assert "HTTP 503" in result[0]["error"]

    def test_request_error_redacts_configured_url(self, monkeypatch):
        secret_base_url = "http://secret-crawl4ai.internal:11235"
        monkeypatch.setenv("CRAWL4AI_URL", secret_base_url)
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        def handler(request: httpx.Request) -> httpx.Response:
            raise httpx.ConnectError("connection refused", request=request)

        with patch("httpx.AsyncClient", side_effect=lambda **_: _mock_async_client(handler)):
            result = asyncio.run(Crawl4AIWebSearchProvider().extract(["https://example.com"]))

        assert "configured CRAWL4AI_URL" in result[0]["error"]
        assert secret_base_url not in result[0]["error"]
        assert "secret-crawl4ai" not in result[0]["error"]

    def test_malformed_json_returns_per_url_error(self, monkeypatch):
        monkeypatch.setenv("CRAWL4AI_URL", "http://crawl4ai:11235")
        from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, text="not-json")

        with patch("httpx.AsyncClient", side_effect=lambda **_: _mock_async_client(handler)):
            result = asyncio.run(Crawl4AIWebSearchProvider().extract(["https://example.com"]))

        assert "parse" in result[0]["error"].lower()


class TestCrawl4AIWebExtractDispatch:
    def test_web_extract_loads_plugins_and_dispatches_to_crawl4ai(self, monkeypatch):
        """Direct web_tools imports should still discover extract-only crawl4ai."""
        from agent.web_search_registry import _reset_for_tests, register_provider
        from tools import web_tools

        _reset_for_tests()
        monkeypatch.setenv("CRAWL4AI_URL", "http://crawl4ai:11235")
        monkeypatch.setattr(web_tools, "_load_web_config", lambda: {"extract_backend": "crawl4ai"})
        async def fake_is_safe_url(url):
            return True

        monkeypatch.setattr(web_tools, "async_is_safe_url", fake_is_safe_url)
        monkeypatch.setattr("tools.interrupt.is_interrupted", lambda: False, raising=False)

        def fake_discover(force=False):
            from plugins.web.crawl4ai.provider import Crawl4AIWebSearchProvider

            register_provider(Crawl4AIWebSearchProvider())

        monkeypatch.setattr("hermes_cli.plugins._ensure_plugins_discovered", fake_discover)

        def handler(request: httpx.Request) -> httpx.Response:
            assert str(request.url) == "http://crawl4ai:11235/md"
            payload = json.loads(request.content.decode())
            assert payload["url"] == "https://example.com"
            assert payload["f"] == "fit"
            return httpx.Response(
                200,
                json={
                    "url": "https://example.com",
                    "filter": "fit",
                    "markdown": "# Example Title\n\nReadable content",
                    "success": True,
                },
            )

        try:
            with patch("plugins.web.crawl4ai.provider.check_website_access", return_value=None), \
                 patch("httpx.AsyncClient", side_effect=lambda **_: _mock_async_client(handler)):
                result = json.loads(asyncio.run(
                    web_tools.web_extract_tool(["https://example.com"], use_llm_processing=False)
                ))
        finally:
            _reset_for_tests()

        assert "results" in result
        assert result["results"][0]["url"] == "https://example.com"
        assert result["results"][0]["title"] == "Example Title"
        assert result["results"][0]["content"] == "# Example Title\n\nReadable content"
