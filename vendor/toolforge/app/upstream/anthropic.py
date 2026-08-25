"""Anthropic Messages upstream transport."""

from __future__ import annotations

from typing import Any, AsyncIterator, Dict, Union

import httpx

from ..config import UpstreamConfig


class AnthropicUpstream:
    def __init__(self, config: UpstreamConfig, *, timeout: float = 180.0) -> None:
        self.config = config
        base = config.base_url.rstrip("/")
        # Accept both https://api.anthropic.com and .../v1
        if base.endswith("/v1"):
            base = base[:-3]
        self._client = httpx.AsyncClient(
            base_url=base,
            timeout=httpx.Timeout(timeout, connect=30.0),
        )

    def _headers(self, *, stream: bool = False) -> Dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self.config.headers.get("anthropic-version", "2023-06-01"),
            **{k: v for k, v in (self.config.headers or {}).items() if k.lower() != "anthropic-version"},
        }
        if self.config.api_key:
            headers["x-api-key"] = self.config.api_key
        if stream:
            headers["Accept"] = "text/event-stream"
        return headers

    async def messages(
        self,
        body: Dict[str, Any],
        *,
        stream: bool = False,
    ) -> Union[Dict[str, Any], AsyncIterator[str]]:
        payload = dict(body)
        payload["stream"] = bool(stream)
        if stream:
            return self._stream(payload)
        response = await self._client.post("/v1/messages", json=payload, headers=self._headers())
        if response.status_code >= 400:
            raise httpx.HTTPStatusError(
                f"upstream {response.status_code}: {response.text[:500]}",
                request=response.request,
                response=response,
            )
        return response.json()

    async def _stream(self, payload: Dict[str, Any]) -> AsyncIterator[str]:
        async with self._client.stream(
            "POST",
            "/v1/messages",
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

    async def aclose(self) -> None:
        await self._client.aclose()
