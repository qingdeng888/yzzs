"""OpenAI-compatible upstream transport."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Optional, Union

import httpx

from ..config import UpstreamConfig
from ..auth import CURRENT_CLIENT_KEY


class OpenAICompatUpstream:
    def __init__(self, config: UpstreamConfig, *, timeout: float = 180.0) -> None:
        self.config = config
        self._timeout = timeout
        self._client = httpx.AsyncClient(
            base_url=config.base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **(self.config.headers or {}),
        }
        key = CURRENT_CLIENT_KEY.get() or self.config.api_key
        if key:
            headers.setdefault("Authorization", f"Bearer {key}")
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    async def chat_completions(
        self,
        body: Dict[str, Any],
        *,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[str]]:
        payload = dict(body)
        payload["stream"] = bool(stream)
        url = "/chat/completions"
        # base_url may already include /v1
        if stream:
            return self._stream(url, payload)
        response = await self._client.post(url, json=payload, headers=self._headers())
        if response.status_code >= 400:
            detail = response.text
            raise httpx.HTTPStatusError(
                f"upstream {response.status_code}: {detail[:500]}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def _stream(self, url: str, payload: Dict[str, Any]) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            url,
            json=payload,
            headers=self._headers(stream=True),
        ) as response:
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise httpx.HTTPStatusError(
                    f"upstream {response.status_code}: {body[:500]}",
                    request=response.request,
                    response=response,
                )
            async for line in response.aiter_lines():
                yield line

    async def list_models(self) -> Dict[str, Any]:
        response = await self._client.get("/models", headers=self._headers())
        if response.status_code >= 400:
            # Fallback: synthesize from config
            return {
                "object": "list",
                "data": [
                    {"id": m, "object": "model", "owned_by": self.config.name}
                    for m in self.config.models
                    if m != "*"
                ],
            }
        return response.json()

    async def aclose(self) -> None:
        await self._client.aclose()
