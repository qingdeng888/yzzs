"""Parse model output into normalized tool calls (Path B)."""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from app.engine.xyml import (
    ProtocolSpec,
    ToolCallConfig,
    ToolSieve,
    openai_tool_calls,
    parse_tool_calls,
)

from ..models.canonical import ToolCall, ToolDef
from .inject import tools_for_sdk

_THINK_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def _config(protocol: str = "XYML") -> ToolCallConfig:
    emit = (protocol or "XYML").strip() or "XYML"
    return ToolCallConfig(
        emit_protocol=emit,
        parse_protocols=[
            ProtocolSpec(emit),
            ProtocolSpec("QNML", parse_only=True),
            ProtocolSpec("XYML", parse_only=True),
        ],
    )


def strip_think_tags(text: str) -> str:
    if not text:
        return ""
    return _THINK_RE.sub("", text)


def parse_text_to_calls(
    text: str,
    tools: List[ToolDef],
    *,
    protocol: str = "XYML",
    strip_think: bool = True,
) -> List[ToolCall]:
    source = strip_think_tags(text) if strip_think else (text or "")
    sdk_tools = tools_for_sdk(tools)
    parsed = parse_tool_calls(source, sdk_tools, config=_config(protocol))
    out: List[ToolCall] = []
    for item in parsed:
        name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
        call_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else "")
        arguments = getattr(item, "input", None)
        if arguments is None and isinstance(item, dict):
            arguments = item.get("input") or item.get("arguments") or {}
        if not isinstance(arguments, dict):
            arguments = {"value": arguments}
        out.append(ToolCall(id=str(call_id or ""), name=str(name or ""), arguments=arguments))
    return out


def to_openai_tool_calls(calls: List[ToolCall]) -> List[Dict[str, Any]]:
    sdk_like = [
        type("C", (), {"id": c.id, "name": c.name, "input": c.arguments})()
        for c in calls
    ]
    # Prefer library formatter when available
    try:
        return openai_tool_calls(sdk_like)  # type: ignore[arg-type]
    except Exception:
        import json

        result = []
        for c in calls:
            result.append(
                {
                    "id": c.id or f"call_{c.name}",
                    "type": "function",
                    "function": {
                        "name": c.name,
                        "arguments": json.dumps(c.arguments or {}, ensure_ascii=False),
                    },
                }
            )
        return result


def create_sieve(tools: List[ToolDef], *, protocol: str = "XYML", hold_length: int = 96) -> ToolSieve:
    return ToolSieve(tools_for_sdk(tools), config=_config(protocol), hold_length=hold_length)


def sieve_events_to_calls(events: List[Dict[str, Any]], tools: List[ToolDef], *, protocol: str = "XYML") -> Tuple[str, List[ToolCall]]:
    content_parts: List[str] = []
    calls: List[ToolCall] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        if event.get("type") == "content":
            content_parts.append(str(event.get("text") or ""))
        elif event.get("type") == "tool_calls":
            raw_calls = event.get("calls") or []
            for item in raw_calls:
                name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
                call_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else "")
                arguments = getattr(item, "input", None)
                if arguments is None and isinstance(item, dict):
                    arguments = item.get("input") or item.get("arguments") or {}
                if not isinstance(arguments, dict):
                    arguments = {"value": arguments}
                calls.append(ToolCall(id=str(call_id or ""), name=str(name or ""), arguments=arguments))
    content = "".join(content_parts)
    if not calls:
        calls = parse_text_to_calls(content, tools, protocol=protocol)
        # If parse found calls inside content, content for client should be empty-ish
        if calls:
            # Keep non-markup prose if any; full parse path already holds markup
            pass
    return content, calls
