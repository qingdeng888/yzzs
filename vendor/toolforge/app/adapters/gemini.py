"""Gemini generateContent client surface."""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException

from ..config import AppConfig
from ..convert import gemini_contents_to_messages, gemini_tools_to_defs
from ..engine.orchestrator import handle_canonical
from ..models.canonical import CanonicalRequest, Message
from ..upstream.router import UpstreamRouter


def body_to_canonical(body: Dict[str, Any], *, model: str, stream: bool) -> CanonicalRequest:
    model = (model or body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    messages = gemini_contents_to_messages(body.get("contents") or [])
    # systemInstruction
    sys_inst = body.get("systemInstruction") or body.get("system_instruction")
    if isinstance(sys_inst, dict):
        parts = sys_inst.get("parts") or []
        texts = [str(p.get("text") or "") for p in parts if isinstance(p, dict)]
        text = "\n".join(t for t in texts if t)
        if text:
            messages = [Message(role="system", content=text)] + messages
    elif isinstance(sys_inst, str) and sys_inst.strip():
        messages = [Message(role="system", content=sys_inst)] + messages

    tools = gemini_tools_to_defs(body.get("tools") or [])
    gen = body.get("generationConfig") or body.get("generation_config") or {}
    return CanonicalRequest(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice="auto",
        stream=stream,
        surface="gemini",
        temperature=gen.get("temperature"),
        max_tokens=gen.get("maxOutputTokens") or gen.get("max_output_tokens"),
        top_p=gen.get("topP") or gen.get("top_p"),
        extra={},
    )


async def handle_generate(
    body: Dict[str, Any],
    *,
    model: str,
    stream: bool,
    config: AppConfig,
    router: UpstreamRouter,
    fc_header: Optional[str] = None,
) -> Any:
    req = body_to_canonical(body, model=model, stream=stream)
    return await handle_canonical(req, config=config, router=router, fc_header=fc_header)
