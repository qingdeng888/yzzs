"""Anthropic Messages client surface."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from ..config import AppConfig
from ..convert import anthropic_messages_to_canonical, anthropic_tools_to_defs
from ..engine.orchestrator import handle_canonical
from ..models.canonical import CanonicalRequest
from ..upstream.router import UpstreamRouter


def body_to_canonical(body: Dict[str, Any]) -> CanonicalRequest:
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")
    messages, _system = anthropic_messages_to_canonical(body)
    tools = anthropic_tools_to_defs(body.get("tools") or [])
    return CanonicalRequest(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=body.get("tool_choice", "auto"),
        stream=bool(body.get("stream")),
        surface="anthropic",
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens"),
        top_p=body.get("top_p"),
        extra={k: body[k] for k in ("metadata", "stop_sequences") if k in body},
    )


async def handle_messages(
    body: Dict[str, Any],
    *,
    config: AppConfig,
    router: UpstreamRouter,
    fc_header: Optional[str] = None,
) -> Any:
    req = body_to_canonical(body)
    return await handle_canonical(req, config=config, router=router, fc_header=fc_header)


async def handle_count_tokens(body: Dict[str, Any]) -> Dict[str, Any]:
    """Cheap estimate stub for SDK compatibility."""
    from ..util.tokens import estimate_tokens

    total = 0
    system = body.get("system")
    if isinstance(system, str):
        total += estimate_tokens(system)
    for msg in body.get("messages") or []:
        if isinstance(msg, dict):
            total += estimate_tokens(str(msg.get("content") or ""))
    for tool in body.get("tools") or []:
        total += estimate_tokens(str(tool))
    return {"input_tokens": total}
