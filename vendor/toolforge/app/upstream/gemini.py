"""Google Gemini generateContent upstream transport."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Union
from urllib.parse import urlencode

import httpx

from ..config import UpstreamConfig


class GeminiUpstream:
    def __init__(self, config: UpstreamConfig, *, timeout: float = 180.0) -> None:
        self.config = config
        base = config.base_url.rstrip("/")
        # Default public API host
        if not base:
            base = "https://generativelanguage.googleapis.com"
        self._client = httpx.AsyncClient(
            base_url=base,
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    def _url(self, model: str, method: str) -> str:
        # method: generateContent | streamGenerateContent
        path = f"/v1beta/models/{model}:{method}"
        if self.config.api_key:
            path = f"{path}?{urlencode({'key': self.config.api_key})}"
        return path

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json", **(self.config.headers or {})}
        return headers

    async def generate_content(
        self,
        model: str,
        body: Dict[str, Any],
        *,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[str]]:
        if stream:
            return self._stream(model, body)
        response = await self._client.post(
            self._url(model, "generateContent"),
            json=body,
            headers=self._headers(),
        )
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"upstream {response.status_code}: {response.text[:500]}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def _stream(self, model: str, body: Dict[str, Any]) -> AsyncIterator[str]:
        # Gemini SSE: alt=sse
        path = self._url(model, "streamGenerateContent")
        sep = "&" if "?" in path else "?"
        path = f"{path}{sep}alt=sse"
        async with self._client.stream("POST", path, json=body, headers=self._headers()) as response:
            if response.status_code >= 400:
                text = (await response.aread()).decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"upstream {response.status_code}: {text[:500]}",
                    request=response.request,
                    response=response,
                )
            async for line in response.aiter_lines():
                yield line

    async def aclose(self) -> None:
        await self._client.aclose()
