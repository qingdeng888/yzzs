"""Model → upstream resolution and client factory."""

from __future__ import annotations

from typing import Dict, Union

from ..config import AppConfig, UpstreamConfig
from .anthropic import AnthropicUpstream
from .gemini import GeminiUpstream
from .openai import OpenAICompatUpstream

UpstreamClientType = Union[OpenAICompatUpstream, AnthropicUpstream, GeminiUpstream]


class UpstreamRouter:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self._clients: Dict[str, UpstreamClientType] = {}

    def resolve(self, model: str) -> tuple[UpstreamConfig, str]:
        return self.config.resolve_model(model)

    def get_client(self, upstream: UpstreamConfig) -> UpstreamClientType:
        client = self._clients.get(upstream.name)
        if client is not None:
            return client

        utype = (upstream.type or "openai_compat").strip().lower()
        timeout = float(self.config.server.timeout_seconds)
        if utype in {"openai_compat", "openai", "grok"}:
            client = OpenAICompatUpstream(upstream, timeout=timeout)
        elif utype == "anthropic":
            client = AnthropicUpstream(upstream, timeout=timeout)
        elif utype == "gemini":
            client = GeminiUpstream(upstream, timeout=timeout)
        else:
            # Best-effort: treat unknown as OpenAI-compatible
            client = OpenAICompatUpstream(upstream, timeout=timeout)

        self._clients[upstream.name] = client
        return client

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
