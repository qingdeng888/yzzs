"""OpenAI Chat Completions client adapter."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from fastapi import HTTPException

from ..config import AppConfig
from ..fc.parse import to_openai_tool_calls
from ..models.canonical import (
    CanonicalRequest,
    ToolCall,
    openai_messages_to_canonical,
    openai_tools_to_defs,
)
from ..upstream.router import UpstreamRouter
from ..util.ids import completion_id, unix_now


PASSTHROUGH_BODY_KEYS = {
    "temperature",
    "top_p",
    "n",
    "stop",
    "max_tokens",
    "max_completion_tokens",
    "presence_penalty",
    "frequency_penalty",
    "logit_bias",
    "user",
    "seed",
    "response_format",
    "logprobs",
    "top_logprobs",
}


def body_to_canonical(body: Dict[str, Any], *, convert_developer: bool) -> CanonicalRequest:
    model = str(body.get("model") or "").strip()
    if not model:
        raise HTTPException(status_code=400, detail="model is required")

    messages = openai_messages_to_canonical(
        body.get("messages") or [],
        convert_developer=convert_developer,
    )
    tools = openai_tools_to_defs(body.get("tools") or body.get("functions") or [])

    return CanonicalRequest(
        model=model,
        messages=messages,
        tools=tools,
        tool_choice=body.get("tool_choice", "auto"),
        stream=bool(body.get("stream")),
        surface="openai",
        temperature=body.get("temperature"),
        max_tokens=body.get("max_tokens") or body.get("max_completion_tokens"),
        top_p=body.get("top_p"),
        extra={k: body[k] for k in PASSTHROUGH_BODY_KEYS if k in body},
    )


def _openai_messages_from_canonical(req: CanonicalRequest) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for msg in req.messages:
        if msg.raw:
            out.append(dict(msg.raw))
            continue
        item: Dict[str, Any] = {"role": msg.role, "content": msg.content}
        if msg.name:
            item["name"] = msg.name
        if msg.tool_call_id:
            item["tool_call_id"] = msg.tool_call_id
        if msg.tool_calls:
            item["tool_calls"] = to_openai_tool_calls(msg.tool_calls)
        out.append(item)
    return out


def build_chat_completion_response(
    *,
    model: str,
    content: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    finish_reason: str = "stop",
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    message: Dict[str, Any] = {"role": "assistant", "content": content or None}
    if tool_calls:
        message["tool_calls"] = to_openai_tool_calls(tool_calls)
        finish_reason = "tool_calls"
        if content == "":
            message["content"] = None
    return {
        "id": completion_id(),
        "object": "chat.completion",
        "created": unix_now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
        "usage": usage
        or {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }


def extract_text_and_native_calls(
    upstream_json: Dict[str, Any],
) -> tuple[str, List[ToolCall], str, Dict[str, int]]:
    choices = upstream_json.get("choices") or []
    if not choices:
        return "", [], "stop", upstream_json.get("usage") or {}
    choice = choices[0] or {}
    message = choice.get("message") or {}
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    finish = str(choice.get("finish_reason") or "stop")
    calls: List[ToolCall] = []
    for tc in message.get("tool_calls") or []:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function") if isinstance(tc.get("function"), dict) else {}
        args = fn.get("arguments")
        parsed: Dict[str, Any]
        if isinstance(args, dict):
            parsed = args
        elif isinstance(args, str) and args.strip():
            try:
                value = json.loads(args)
                parsed = value if isinstance(value, dict) else {"value": value}
            except Exception:
                parsed = {"_raw": args}
        else:
            parsed = {}
        calls.append(
            ToolCall(
                id=str(tc.get("id") or ""),
                name=str(fn.get("name") or ""),
                arguments=parsed,
            )
        )
    usage = upstream_json.get("usage") if isinstance(upstream_json.get("usage"), dict) else {}
    return content, calls, finish, usage  # type: ignore[return-value]


async def handle_chat_completions(
    body: Dict[str, Any],
    *,
    config: AppConfig,
    router: UpstreamRouter,
    fc_header: Optional[str] = None,
) -> Any:
    from ..engine.orchestrator import handle_canonical

    req = body_to_canonical(
        body,
        convert_developer=config.features.convert_developer_to_system,
    )
    return await handle_canonical(req, config=config, router=router, fc_header=fc_header)
