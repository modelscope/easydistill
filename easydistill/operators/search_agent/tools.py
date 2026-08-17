# Copyright 2026 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Search tools for the search-agent pipeline: LLM-simulated and real web."""

import json
import logging
import os
import re
import sqlite3
import threading
from typing import Any, Dict, List, Optional, cast

from easydistill.backends.base import ModelBackend

from .utils import ROLE_SEARCH_SIM, call_role, parse_json_safely

logger = logging.getLogger(__name__)

# Tool schemas exposed to the solver, identical to the original SEARCH_TOOLS.
SEARCH_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "web_search",
        "description": "Perform a web search to find information about a query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "The search query."}},
            "required": ["query"],
        },
    },
    {
        "name": "web_browse",
        "description": "Visit a specific URL to read its content.",
        "parameters": {
            "type": "object",
            "properties": {"url": {"type": "string", "description": "The URL to visit."}},
            "required": ["url"],
        },
    },
]

MOCK_SEARCH_SYSTEM_PROMPT = """You are Google Search, a web search engine.

Given a user search query, you return a small list of relevant web search
results in JSON format. Each result should look like what a typical web search
API would return.

You MUST respond with **only** a single JSON object and nothing else.
The JSON schema is:
{
  "results": [
    {
      "title": "<page title>",
      "url": "<https URL>",
      "snippet": "<short text snippet showing why this page is relevant>"
    },
    ...
  ]
}

- "title" should be concise but descriptive.
- "url" should be a plausible HTTPS URL.
- "snippet" should briefly summarize the relevant part of the page.

If you know the answer, it should be clearly reflected in your response.
You must provide true information! DO NOT fabricate.
"""

MOCK_BROWSER_SYSTEM_PROMPT = """You are a mock web browser and content fetcher.

Given a URL (and optionally some search context like the original query and
snippet), you simulate visiting the page and returning its title and main
body text.

You MUST respond with **only** a single JSON object and nothing else.
The JSON schema is:
{
  "url": "<the same url you received>",
  "title": "<page title>",
  "content": "<main body text of the page>"
}

- "content" should be a coherent, self-contained article-like text.
- If a ``max_chars`` limit is provided, you should truncate ``content`` to
  at most that many characters and, if truncated, append "...[TRUNCATED]".
"""


def mock_web_search(
    backend: ModelBackend,
    config: Dict[str, Any],
    query: str,
    num_results: int = 5,
    long_snippet: bool = False,
) -> Dict[str, Any]:
    """LLM-simulated web search returning ``{"results": [...]}``."""
    if num_results <= 0:
        return {"results": []}
    user_prompt = (
        f"Search query: {query}\n\nReturn up to {num_results} results in the JSON "
        "format described above."
    )
    if long_snippet:
        user_prompt += (
            "\n\nIMPORTANT: Each result snippet must be a single long paragraph "
            "(>= 300 characters) with multiple sentences, containing rich factual "
            "detail related to the query."
        )
    content = call_role(backend, config, ROLE_SEARCH_SIM, user_prompt, MOCK_SEARCH_SYSTEM_PROMPT)
    data = parse_json_safely(content)
    if not isinstance(data, dict) or not isinstance(data.get("results"), list):
        raise ValueError(f"mock_web_search returned unparseable output: {content[:200]}")
    return data


def mock_web_browse(
    backend: ModelBackend,
    config: Dict[str, Any],
    url: str,
    max_chars: int = 2000,
    query: Optional[str] = None,
    snippet: Optional[str] = None,
) -> Dict[str, Any]:
    """LLM-simulated page fetch returning ``{"url","title","content"}``."""
    if not url:
        raise ValueError("url must not be empty")
    context_lines = [f"URL: {url}"]
    if query:
        context_lines.append(f"Search query: {query}")
    if snippet:
        context_lines.append(f"Search snippet: {snippet}")
    context_lines.append(f"Max content characters: {max_chars}")
    user_prompt = "\n".join(context_lines) + (
        "\n\nReturn a JSON object in the schema described above."
    )
    content = call_role(backend, config, ROLE_SEARCH_SIM, user_prompt, MOCK_BROWSER_SYSTEM_PROMPT)
    data = parse_json_safely(content)
    if not isinstance(data, dict):
        raise ValueError(f"mock_web_browse returned unparseable output: {content[:200]}")
    result = {
        "url": str(data.get("url", url)),
        "title": str(data.get("title", "")),
        "content": str(data.get("content", "")),
    }
    if max_chars > 0 and len(result["content"]) > max_chars:
        result["content"] = result["content"][:max_chars] + "...[TRUNCATED]"
    return result


class SqliteCache:
    """Thread-safe SQLite key-value cache for real search/browse results."""

    def __init__(self, db_path: str):
        self._db_path = db_path
        self._lock = threading.Lock()
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        with self._connect() as conn:
            conn.execute("CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT)")

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self._db_path, timeout=30)

    def get(self, key: str) -> Optional[str]:
        with self._lock, self._connect() as conn:
            row = conn.execute("SELECT value FROM cache WHERE key = ?", (key,)).fetchone()
        return row[0] if row else None

    def set(self, key: str, value: str) -> None:
        with self._lock, self._connect() as conn:
            conn.execute("INSERT OR REPLACE INTO cache (key, value) VALUES (?, ?)", (key, value))


def real_web_search(
    config: Dict[str, Any],
    query: str,
    num_results: int = 5,
    cache: Optional[SqliteCache] = None,
) -> Dict[str, Any]:
    """Real web search via the Google Custom Search JSON API.

    Requires ``tools.google_api_key`` (or env GOOGLE_API_KEY) and
    ``tools.google_cx`` (or env GOOGLE_CX). Results are cached when a cache
    is provided.
    """
    import requests

    tools_cfg = config.get("tools") or {}
    cache_key = f"search::{query}::{num_results}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return cast(Dict[str, Any], json.loads(cached))

    api_key = tools_cfg.get("google_api_key") or os.getenv("GOOGLE_API_KEY", "")
    cx = tools_cfg.get("google_cx") or os.getenv("GOOGLE_CX", "")
    if not api_key:
        raise ValueError("Real web search requires tools.google_api_key or GOOGLE_API_KEY")
    if not cx:
        raise ValueError("Real web search requires tools.google_cx or GOOGLE_CX")
    resp = requests.get(
        "https://www.googleapis.com/customsearch/v1",
        params={"key": api_key, "cx": cx, "q": query, "num": min(num_results, 10)},
        timeout=30,
    )
    resp.raise_for_status()
    payload = resp.json()
    results = [
        {
            "title": item.get("title", ""),
            "url": item.get("link", ""),
            "snippet": item.get("snippet", ""),
        }
        for item in payload.get("items", [])[:num_results]
    ]
    data = {"results": results}
    if cache:
        cache.set(cache_key, json.dumps(data, ensure_ascii=False))
    return data


def real_web_browse(
    config: Dict[str, Any],
    url: str,
    max_chars: int = 8000,
    cache: Optional[SqliteCache] = None,
) -> Dict[str, Any]:
    """Real page fetch via the Jina Reader API (plain HTTP, no browser).

    ``tools.jina_api_key`` (or env JINA_API_KEY) is optional and only raises
    the rate limit when provided.
    """
    import requests

    if not url:
        raise ValueError("url must not be empty")
    if not (url.startswith("http://") or url.startswith("https://")):
        raise ValueError(f"real_web_browse requires an HTTP(S) URL, got {url!r}")
    cache_key = f"browse::{url}::{max_chars}"
    if cache:
        cached = cache.get(cache_key)
        if cached:
            return cast(Dict[str, Any], json.loads(cached))

    tools_cfg = config.get("tools") or {}
    headers = {}
    jina_key = tools_cfg.get("jina_api_key") or os.getenv("JINA_API_KEY", "")
    if jina_key:
        headers["Authorization"] = f"Bearer {jina_key}"
    resp = requests.get(f"https://r.jina.ai/{url}", headers=headers, timeout=60)
    resp.raise_for_status()
    text = resp.text
    title = ""
    title_match = re.search(r"^Title:\s*(.+)$", text, re.MULTILINE)
    if title_match:
        title = title_match.group(1).strip()
    content = text
    if max_chars > 0 and len(content) > max_chars:
        content = content[:max_chars] + "...[TRUNCATED]"
    data = {"url": url, "title": title, "content": content}
    if cache:
        cache.set(cache_key, json.dumps(data, ensure_ascii=False))
    return data


class SearchToolset:
    """Dispatch web_search/web_browse calls in mock or real mode.

    In mock mode the tools are simulated by the ``search_sim`` role of the
    backend. In real mode they hit the Google/Jina APIs with an optional
    SQLite cache (``tools.cache_db_path``).
    """

    def __init__(self, backend: ModelBackend, config: Dict[str, Any]):
        self.backend = backend
        self.config = config
        tools_cfg = config.get("tools") or {}
        self.mode = str(tools_cfg.get("mode", "mock")).lower()
        self.num_results = int(tools_cfg.get("num_results", 5))
        # Default 500 chars matches the original mock_fetch_page default.
        self.browse_max_chars = int(tools_cfg.get("browse_max_chars", 500))
        self._cache: Optional[SqliteCache] = None
        cache_path = tools_cfg.get("cache_db_path")
        if self.mode == "real" and cache_path:
            self._cache = SqliteCache(str(cache_path))

    def search(self, query: str, long_snippet: bool = False) -> Dict[str, Any]:
        if self.mode == "real":
            return real_web_search(self.config, query, self.num_results, self._cache)
        return mock_web_search(self.backend, self.config, query, self.num_results, long_snippet)

    def browse(
        self,
        url: str,
        query: Optional[str] = None,
        snippet: Optional[str] = None,
    ) -> Dict[str, Any]:
        if self.mode == "real":
            return real_web_browse(self.config, url, self.browse_max_chars, self._cache)
        return mock_web_browse(
            self.backend, self.config, url, self.browse_max_chars, query, snippet
        )

    def execute(
        self,
        tool_name: Optional[str],
        arguments: Dict[str, Any],
        solve_history: List[Dict[str, Any]],
    ) -> str:
        """Execute a solver tool call and return the tool response text."""
        if tool_name == "web_search":
            query = arguments.get("query")
            if not query:
                return "Error: Missing 'query' argument for web_search."
            result = self.search(query)
            return json.dumps(result, ensure_ascii=False)
        if tool_name == "web_browse":
            url = arguments.get("url")
            if not url:
                return "Error: Missing 'url' argument for web_browse."
            query, snippet = _find_browse_context(solve_history, url)
            result = self.browse(url, query=query, snippet=snippet)
            return json.dumps(result, ensure_ascii=False)
        return f"Error: Unknown tool '{tool_name}'."


def _find_browse_context(solve_history: List[Dict[str, Any]], url: str) -> tuple:
    """Recover the search query/snippet that produced a URL from the history.

    Mirrors the original mock-browse context recovery so the simulator stays
    consistent with earlier search results.
    """
    snippet_found = None
    for msg in reversed(solve_history or []):
        if msg.get("role") == "user" and "<tool_response>" in msg.get("content", ""):
            try:
                resp_json_str = (
                    msg["content"]
                    .replace("<tool_response>", "")
                    .replace("</tool_response>", "")
                    .strip()
                )
                resp_data = json.loads(resp_json_str)
            except (json.JSONDecodeError, KeyError):
                continue
            for result in resp_data.get("results", []) or []:
                if result.get("url") == url:
                    snippet_found = result.get("snippet")
                    break
            if snippet_found:
                break

    query_found = None
    if snippet_found:
        for msg in reversed(solve_history or []):
            if msg.get("role") == "assistant" and "web_search" in msg.get("content", ""):
                q_match = re.search(r'"query":\s*"([^"]+)"', msg["content"])
                if q_match:
                    query_found = q_match.group(1)
                    break
    return query_found, snippet_found
