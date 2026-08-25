"""Canonical request → upstream wire format (openai/anthropic/gemini)."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from ..convert import (
    anthropic_response,
    canonical_to_anthropic_messages,
    gemini_response,
    messages_to_gemini_contents,
    tools_to_anthropic,
    tools_to_gemini,
)
from ..fc.inject import inject_prompt_messages, strip_tools_from_openai_body
from ..fc.parse import parse_text_to_calls, to_openai_tool_calls
from ..models.canonical import CanonicalRequest, CanonicalResponse, ToolCall, tool_defs_to_openai
from ..adapters.openai_chat import (
    _openai_messages_from_canonical,
    build_chat_completion_response,
    extract_text_and_native_calls,
)


def build_openai_wire(
    req: CanonicalRequest,
    *,
    upstream_model: str,
    protocol: str,
    convert_developer: bool,
    instructions_extra: str = "",
) -> Dict[str, Any]:
    body: Dict[str, Any] = {"model": upstream_model, "stream": req.stream}
    body.update(req.extra or {})
    if req.fc_mode == "prompt":
        messages = inject_prompt_messages(
            req.messages,
            req.tools,
            protocol=protocol,
            convert_developer_to_system=convert_developer,
        )
        if instructions_extra:
            if messages and messages[0].get("role") == "system":
                messages[0]["content"] = str(messages[0].get("content") or "") + "\n\n" + instructions_extra
            else:
                messages = [{"role": "system", "content": instructions_extra}] + messages
        body["messages"] = messages
        return strip_tools_from_openai_body(body)

    body["messages"] = _openai_messages_from_canonical(req)
    if req.fc_mode == "native" and req.tools:
        body["tools"] = tool_defs_to_openai(req.tools)
        if req.tool_choice is not None:
            body["tool_choice"] = req.tool_choice
    return body


def build_anthropic_wire(
    req: CanonicalRequest,
    *,
    upstream_model: str,
    protocol: str,
    max_tokens: int = 4096,
    instructions_extra: str = "",
) -> Dict[str, Any]:
    body: Dict[str, Any] = {
        "model": upstream_model,
        "max_tokens": int(req.max_tokens or max_tokens),
        "stream": req.stream,
    }
    if req.temperature is not None:
        body["temperature"] = req.temperature
    if req.top_p is not None:
        body["top_p"] = req.top_p

    if req.fc_mode == "prompt":
        oai_messages = inject_prompt_messages(req.messages, req.tools, protocol=protocol)
        if instructions_extra:
            if oai_messages and oai_messages[0].get("role") == "system":
                oai_messages[0]["content"] = str(oai_messages[0].get("content") or "") + "\n\n" + instructions_extra
            else:
                oai_messages.insert(0, {"role": "system", "content": instructions_extra})
        system_parts = []
        messages = []
        for m in oai_messages:
            if m.get("role") == "system":
                system_parts.append(str(m.get("content") or ""))
            else:
                role = m.get("role") if m.get("role") in {"user", "assistant"} else "user"
                messages.append({"role": role, "content": m.get("content") or ""})
        if system_parts:
            body["system"] = "\n\n".join(system_parts)
        body["messages"] = messages or [{"role": "user", "content": "Hello"}]
        return body

    system, messages = canonical_to_anthropic_messages(req.messages)
    if system:
        body["system"] = system
    body["messages"] = messages or [{"role": "user", "content": "Hello"}]
    if req.fc_mode == "native" and req.tools:
        body["tools"] = tools_to_anthropic(req.tools)
        # tool_choice mapping
        tc = req.tool_choice
        if tc == "auto" or tc is None:
            body["tool_choice"] = {"type": "auto"}
        elif tc == "none":
            body["tool_choice"] = {"type": "none"}
        elif tc == "required" or tc == "any":
            body["tool_choice"] = {"type": "any"}
        elif isinstance(tc, dict) and tc.get("type") == "function":
            name = (tc.get("function") or {}).get("name")
            if name:
                body["tool_choice"] = {"type": "tool", "name": name}
    return body


def build_gemini_wire(
    req: CanonicalRequest,
    *,
    protocol: str,
    instructions_extra: str = "",
) -> Dict[str, Any]:
    body: Dict[str, Any] = {}
    gen_cfg: Dict[str, Any] = {}
    if req.temperature is not None:
        gen_cfg["temperature"] = req.temperature
    if req.top_p is not None:
        gen_cfg["topP"] = req.top_p
    if req.max_tokens is not None:
        gen_cfg["maxOutputTokens"] = req.max_tokens
    if gen_cfg:
        body["generationConfig"] = gen_cfg

    if req.fc_mode == "prompt":
        oai_messages = inject_prompt_messages(req.messages, req.tools, protocol=protocol)
        if instructions_extra:
            if oai_messages and oai_messages[0].get("role") == "system":
                oai_messages[0]["content"] = str(oai_messages[0].get("content") or "") + "\n\n" + instructions_extra
            else:
                oai_messages.insert(0, {"role": "system", "content": instructions_extra})
        contents = []
        system_chunks = []
        for m in oai_messages:
            role = m.get("role")
            text = str(m.get("content") or "")
            if role == "system":
                system_chunks.append(text)
                continue
            g_role = "model" if role == "assistant" else "user"
            contents.append({"role": g_role, "parts": [{"text": text}]})
        if system_chunks:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_chunks)}]}
        body["contents"] = contents or [{"role": "user", "parts": [{"text": "Hello"}]}]
        return body

    # native / passthrough
    system_parts = [m.content for m in req.messages if m.role == "system" and isinstance(m.content, str)]
    non_system = [m for m in req.messages if m.role != "system"]
    body["contents"] = messages_to_gemini_contents(non_system) or [
        {"role": "user", "parts": [{"text": "Hello"}]}
    ]
    if system_parts:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
    if req.fc_mode == "native" and req.tools:
        body["tools"] = tools_to_gemini(req.tools)
    return body


def openai_result_to_canonical(
    result: Dict[str, Any],
    *,
    req: CanonicalRequest,
    protocol: str,
    strip_think: bool,
) -> CanonicalResponse:
    content, native_calls, finish, usage = extract_text_and_native_calls(result)
    if req.fc_mode == "prompt":
        calls = parse_text_to_calls(content, req.tools, protocol=protocol, strip_think=strip_think)
        if not calls and native_calls:
            calls = native_calls
        return CanonicalResponse(
            model=req.model,
            content="" if calls else content,
            tool_calls=calls,
            finish_reason="tool_calls" if calls else finish,
            usage=usage or {},
            raw=result,
        )
    return CanonicalResponse(
        model=req.model,
        content=content,
        tool_calls=native_calls,
        finish_reason=finish,
        usage=usage or {},
        raw=result,
    )


def anthropic_result_to_canonical(
    result: Dict[str, Any],
    *,
    req: CanonicalRequest,
    protocol: str,
    strip_think: bool,
) -> CanonicalResponse:
    blocks = result.get("content") or []
    text_parts: List[str] = []
    calls: List[ToolCall] = []
    for block in blocks:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text_parts.append(str(block.get("text") or ""))
        elif block.get("type") == "tool_use":
            args = block.get("input") if isinstance(block.get("input"), dict) else {}
            calls.append(
                ToolCall(
                    id=str(block.get("id") or ""),
                    name=str(block.get("name") or ""),
                    arguments=args,
                )
            )
    text = "".join(text_parts)
    usage_raw = result.get("usage") if isinstance(result.get("usage"), dict) else {}
    usage = {
        "prompt_tokens": usage_raw.get("input_tokens") or 0,
        "completion_tokens": usage_raw.get("output_tokens") or 0,
        "total_tokens": (usage_raw.get("input_tokens") or 0) + (usage_raw.get("output_tokens") or 0),
    }
    if req.fc_mode == "prompt":
        parsed = parse_text_to_calls(text, req.tools, protocol=protocol, strip_think=strip_think)
        if parsed:
            calls = parsed
            text = ""
    stop = str(result.get("stop_reason") or "end_turn")
    finish = "tool_calls" if calls else ("stop" if stop == "end_turn" else stop)
    return CanonicalResponse(
        model=req.model,
        content=text,
        tool_calls=calls,
        finish_reason=finish,
        usage=usage,
        raw=result,
    )


def gemini_result_to_canonical(
    result: Dict[str, Any],
    *,
    req: CanonicalRequest,
    protocol: str,
    strip_think: bool,
) -> CanonicalResponse:
    candidates = result.get("candidates") or []
    text_parts: List[str] = []
    calls: List[ToolCall] = []
    if candidates:
        content = (candidates[0] or {}).get("content") or {}
        for part in content.get("parts") or []:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                text_parts.append(str(part.get("text") or ""))
            fc = part.get("functionCall") or part.get("function_call")
            if isinstance(fc, dict):
                args = fc.get("args") or {}
                if not isinstance(args, dict):
                    args = {"value": args}
                calls.append(ToolCall(id="", name=str(fc.get("name") or ""), arguments=args))
    text = "".join(text_parts)
    if req.fc_mode == "prompt":
        parsed = parse_text_to_calls(text, req.tools, protocol=protocol, strip_think=strip_think)
        if parsed:
            calls = parsed
            text = ""
    meta = result.get("usageMetadata") if isinstance(result.get("usageMetadata"), dict) else {}
    usage = {
        "prompt_tokens": meta.get("promptTokenCount") or 0,
        "completion_tokens": meta.get("candidatesTokenCount") or 0,
        "total_tokens": meta.get("totalTokenCount") or 0,
    }
    return CanonicalResponse(
        model=req.model,
        content=text,
        tool_calls=calls,
        finish_reason="tool_calls" if calls else "stop",
        usage=usage,
        raw=result,
    )


def encode_client_response(req: CanonicalRequest, resp: CanonicalResponse) -> Dict[str, Any]:
    if req.surface == "anthropic":
        return anthropic_response(
            model=resp.model,
            content=resp.content,
            tool_calls=resp.tool_calls or None,
            usage=resp.usage,
        )
    if req.surface == "gemini":
        return gemini_response(
            model=resp.model,
            content=resp.content,
            tool_calls=resp.tool_calls or None,
        )
    if req.surface == "responses":
        from ..convert import build_responses_response

        return build_responses_response(
            model=resp.model,
            content=resp.content,
            tool_calls=resp.tool_calls or None,
            usage=resp.usage,
        )
    return build_chat_completion_response(
        model=resp.model,
        content=resp.content,
        tool_calls=resp.tool_calls or None,
        finish_reason=resp.finish_reason,
        usage=resp.usage,
    )
