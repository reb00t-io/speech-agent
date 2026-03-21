from __future__ import annotations

import re
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup

DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_FETCH_CHARS = 8000
WEB_SEARCH_USAGE_INSTRUCTIONS = (
    "Use these search results as pointers: extract only the information relevant to the user's request, "
    "open links with fetch_url when needed for verification or deeper details, summarize what was asked, "
    "and do not expose unfiltered raw search output to the user."
)


def normalize_max_results(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_MAX_RESULTS
    if isinstance(value, (int, float)):
        return max(1, min(10, int(value)))
    return DEFAULT_MAX_RESULTS


def normalize_max_chars(value: Any) -> int:
    if isinstance(value, bool):
        return DEFAULT_MAX_FETCH_CHARS
    if isinstance(value, (int, float)):
        return max(500, min(20000, int(value)))
    return DEFAULT_MAX_FETCH_CHARS


def normalize_fetch_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.netloc == "duckduckgo.com" and parsed.path == "/l/":
        params = parse_qs(parsed.query)
        target = params.get("uddg", [""])[0]
        if target:
            return unquote(target)
    if url.startswith("//"):
        return f"https:{url}"
    return url


def flatten_related_topics(related_topics: list[dict[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for topic in related_topics:
        if isinstance(topic.get("Topics"), list):
            for item in topic.get("Topics") or []:
                if isinstance(item, dict):
                    flattened.append(item)
            continue
        flattened.append(topic)
    return flattened


def extract_text_title(text: str) -> tuple[str, str]:
    if " - " in text:
        title, snippet = text.split(" - ", 1)
        return title.strip(), snippet.strip()
    return text.strip(), text.strip()


def parse_duckduckgo_html_results(html: str, max_results: int) -> list[dict[str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    parsed_results: list[dict[str, str]] = []
    for result in soup.select(".result"):
        link_el = result.select_one("a.result__a")
        if not link_el:
            continue

        title = link_el.get_text(" ", strip=True)
        url = str(link_el.get("href") or "").strip()
        snippet_el = result.select_one(".result__snippet")
        snippet = snippet_el.get_text(" ", strip=True) if snippet_el else ""
        if not title and not url:
            continue
        source = (urlparse(url).netloc or "duckduckgo").lower() if url else "duckduckgo"
        parsed_results.append({
            "title": title or url,
            "snippet": snippet,
            "url": url,
            "source": source,
        })
        if len(parsed_results) >= max_results:
            break
    return parsed_results


async def _web_search_html(session: aiohttp.ClientSession, query: str, max_results: int) -> list[dict[str, str]]:
    try:
        async with session.get(
            "https://html.duckduckgo.com/html/",
            params={"q": query},
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            },
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            resp.raise_for_status()
            return parse_duckduckgo_html_results(await resp.text(), max_results)
    except Exception:
        return []


async def web_search(session: aiohttp.ClientSession, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> dict[str, Any]:
    max_results = normalize_max_results(max_results)
    html_results = await _web_search_html(session, query, max_results)
    if html_results:
        return {
            "query": query,
            "results": html_results,
            "usage_instructions": WEB_SEARCH_USAGE_INSTRUCTIONS,
        }

    try:
        async with session.get(
            "https://api.duckduckgo.com/",
            params={
                "q": query,
                "format": "json",
                "no_html": 1,
                "no_redirect": 1,
                "skip_disambig": 1,
            },
            timeout=aiohttp.ClientTimeout(total=20),
        ) as resp:
            resp.raise_for_status()
            payload = await resp.json(content_type=None)
    except Exception as exc:
        return {"query": query, "error": f"Web search failed: {type(exc).__name__}: {exc}"}

    results: list[dict[str, str]] = []
    abstract = (payload.get("AbstractText") or "").strip()
    abstract_url = (payload.get("AbstractURL") or "").strip()
    heading = (payload.get("Heading") or query).strip()
    if abstract:
        results.append({
            "title": heading,
            "snippet": abstract,
            "url": abstract_url,
            "source": (urlparse(abstract_url).netloc or "duckduckgo").lower() if abstract_url else "duckduckgo",
        })

    for item in flatten_related_topics(payload.get("RelatedTopics") or []):
        text = (item.get("Text") or "").strip()
        first_url = (item.get("FirstURL") or "").strip()
        if not text and not first_url:
            continue
        title, snippet = extract_text_title(text) if text else (first_url, "")
        results.append({
            "title": title,
            "snippet": snippet,
            "url": first_url,
            "source": (urlparse(first_url).netloc or "duckduckgo").lower() if first_url else "duckduckgo",
        })
        if len(results) >= max_results:
            break

    return {
        "query": query,
        "results": results[:max_results],
        "usage_instructions": WEB_SEARCH_USAGE_INSTRUCTIONS,
    }


def _html_to_text(html: str) -> tuple[str, str]:
    soup = BeautifulSoup(html, "html.parser")
    title = (soup.title.string or "").strip() if soup.title and soup.title.string else ""
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
    return title, text


async def fetch_url(session: aiohttp.ClientSession, url: str, max_chars: int = DEFAULT_MAX_FETCH_CHARS) -> dict[str, Any]:
    resolved_url = normalize_fetch_url((url or "").strip())
    parsed = urlparse(resolved_url)
    if parsed.scheme not in {"http", "https"}:
        return {"url": resolved_url, "error": "Only http/https URLs are supported"}

    max_chars = normalize_max_chars(max_chars)
    try:
        async with session.get(
            resolved_url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/122.0.0.0 Safari/537.36"
                ),
            },
            timeout=aiohttp.ClientTimeout(total=30),
            allow_redirects=True,
        ) as resp:
            resp.raise_for_status()
            content_type = (resp.headers.get("Content-Type") or "").lower()
            body = await resp.text(errors="ignore")
            final_url = str(resp.url)
    except Exception as exc:
        return {"url": resolved_url, "error": f"fetch_url failed: {type(exc).__name__}: {exc}"}

    title = ""
    text = body
    if "html" in content_type:
        title, text = _html_to_text(body)
    else:
        text = re.sub(r"\s+", " ", body).strip()

    truncated = text[:max_chars]
    return {
        "url": resolved_url,
        "final_url": final_url,
        "title": title,
        "content_type": content_type,
        "text": truncated,
        "truncated": len(text) > len(truncated),
        "text_length": len(text),
    }
