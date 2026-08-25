"""Canonical ↔ protocol shape converters."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .fc.parse import to_openai_tool_calls
from .models.canonical import Message, ToolCall, ToolDef
from .util.ids import call_id, completion_id, random_id, unix_now


# ---------- Anthropic ----------

def anthropic_tools_to_defs(raw_tools: Any) -> List[ToolDef]:
    defs: List[ToolDef] = []
    for item in raw_tools or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("name") or "")
        if not name:
            continue
        schema = item.get("input_schema") or item.get("parameters") or {"type": "object", "properties": {}}
        defs.append(
            ToolDef(
                name=name,
                description=str(item.get("description") or ""),
                parameters=schema if isinstance(schema, dict) else {},
                raw=item,
            )
        )
    return defs


def tools_to_anthropic(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    out = []
    for t in tools:
        out.append(
            {
                "name": t.name,
                "description": t.description or "",
                "input_schema": t.parameters or {"type": "object", "properties": {}},
            }
        )
    return out


def anthropic_messages_to_canonical(body: Dict[str, Any]) -> Tuple[List[Message], str]:
    """Return (messages, system_text)."""
    system = body.get("system") or ""
    if isinstance(system, list):
        parts = []
        for block in system:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        system = "\n".join(parts)
    system = str(system or "")

    messages: List[Message] = []
    if system.strip():
        messages.append(Message(role="system", content=system))

    for item in body.get("messages") or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        content = item.get("content")
        tool_calls: List[ToolCall] = []
        text_parts: List[str] = []
        tool_results: List[Message] = []

        if isinstance(content, str):
            text_parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if not isinstance(block, dict):
                    continue
                btype = block.get("type")
                if btype == "text":
                    text_parts.append(str(block.get("text") or ""))
                elif btype == "tool_use":
                    tool_calls.append(
                        ToolCall(
                            id=str(block.get("id") or call_id()),
                            name=str(block.get("name") or ""),
                            arguments=block.get("input") if isinstance(block.get("input"), dict) else {},
                        )
                    )
                elif btype == "tool_result":
                    tool_results.append(
                        Message(
                            role="tool",
                            content=_block_to_text(block.get("content")),
                            tool_call_id=str(block.get("tool_use_id") or ""),
                            raw=block,
                        )
                    )

        if role == "assistant":
            messages.append(
                Message(
                    role="assistant",
                    content="\n".join(text_parts),
                    tool_calls=tool_calls,
                    raw=item,
                )
            )
        elif role == "user":
            if text_parts:
                messages.append(Message(role="user", content="\n".join(text_parts), raw=item))
            for tr in tool_results:
                messages.append(tr)
        else:
            messages.append(Message(role=role, content="\n".join(text_parts), raw=item))

    return messages, system


def _block_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for b in content:
            if isinstance(b, dict) and b.get("type") == "text":
                parts.append(str(b.get("text") or ""))
            elif isinstance(b, str):
                parts.append(b)
        return "\n".join(parts)
    return str(content)


def canonical_to_anthropic_messages(messages: List[Message]) -> Tuple[str, List[Dict[str, Any]]]:
    system_parts: List[str] = []
    out: List[Dict[str, Any]] = []
    pending_tool_results: List[Dict[str, Any]] = []

    def flush_tool_results():
        nonlocal pending_tool_results
        if pending_tool_results:
            out.append({"role": "user", "content": pending_tool_results})
            pending_tool_results = []

    for msg in messages:
        if msg.role == "system":
            if isinstance(msg.content, str) and msg.content.strip():
                system_parts.append(msg.content)
            continue
        if msg.role == "tool":
            pending_tool_results.append(
                {
                    "type": "tool_result",
                    "tool_use_id": msg.tool_call_id or "",
                    "content": msg.content if isinstance(msg.content, str) else json.dumps(msg.content),
                }
            )
            continue
        flush_tool_results()
        if msg.role == "assistant":
            blocks: List[Dict[str, Any]] = []
            text = msg.content if isinstance(msg.content, str) else ""
            if text:
                blocks.append({"type": "text", "text": text})
            for call in msg.tool_calls:
                blocks.append(
                    {
                        "type": "tool_use",
                        "id": call.id or call_id(),
                        "name": call.name,
                        "input": call.arguments or {},
                    }
                )
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})
        else:
            out.append({"role": "user", "content": msg.content if isinstance(msg.content, str) else str(msg.content)})
    flush_tool_results()
    return "\n\n".join(system_parts), out


def anthropic_response(
    *,
    model: str,
    content: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    stop_reason: str = "end_turn",
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    blocks: List[Dict[str, Any]] = []
    if content:
        blocks.append({"type": "text", "text": content})
    if tool_calls:
        for call in tool_calls:
            blocks.append(
                {
                    "type": "tool_use",
                    "id": call.id or call_id(),
                    "name": call.name,
                    "input": call.arguments or {},
                }
            )
        stop_reason = "tool_use"
    if not blocks:
        blocks = [{"type": "text", "text": ""}]
    return {
        "id": f"msg_{random_id(20)}",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": (usage or {}).get("prompt_tokens") or (usage or {}).get("input_tokens") or 0,
            "output_tokens": (usage or {}).get("completion_tokens") or (usage or {}).get("output_tokens") or 0,
        },
    }


# ---------- Gemini ----------

def gemini_tools_to_defs(raw: Any) -> List[ToolDef]:
    defs: List[ToolDef] = []
    for tool in raw or []:
        if not isinstance(tool, dict):
            continue
        decls = tool.get("functionDeclarations") or tool.get("function_declarations") or []
        for decl in decls:
            if not isinstance(decl, dict):
                continue
            name = str(decl.get("name") or "")
            if not name:
                continue
            params = decl.get("parameters") or {"type": "object", "properties": {}}
            defs.append(
                ToolDef(
                    name=name,
                    description=str(decl.get("description") or ""),
                    parameters=params if isinstance(params, dict) else {},
                    raw=decl,
                )
            )
    return defs


def tools_to_gemini(tools: List[ToolDef]) -> List[Dict[str, Any]]:
    decls = []
    for t in tools:
        decls.append(
            {
                "name": t.name,
                "description": t.description or "",
                "parameters": t.parameters or {"type": "object", "properties": {}},
            }
        )
    return [{"functionDeclarations": decls}] if decls else []


def gemini_contents_to_messages(contents: Any) -> List[Message]:
    messages: List[Message] = []
    for item in contents or []:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role") or "user")
        if role == "model":
            role = "assistant"
        parts = item.get("parts") or []
        text_parts: List[str] = []
        tool_calls: List[ToolCall] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            if "text" in part:
                text_parts.append(str(part.get("text") or ""))
            fc = part.get("functionCall") or part.get("function_call")
            if isinstance(fc, dict):
                args = fc.get("args") or fc.get("arguments") or {}
                if not isinstance(args, dict):
                    args = {"value": args}
                tool_calls.append(
                    ToolCall(
                        id=call_id(),
                        name=str(fc.get("name") or ""),
                        arguments=args,
                    )
                )
            fr = part.get("functionResponse") or part.get("function_response")
            if isinstance(fr, dict):
                messages.append(
                    Message(
                        role="tool",
                        name=str(fr.get("name") or ""),
                        content=json.dumps(fr.get("response") or {}, ensure_ascii=False),
                        tool_call_id=call_id(),
                    )
                )
        if role == "assistant" and (text_parts or tool_calls):
            messages.append(
                Message(role="assistant", content="\n".join(text_parts), tool_calls=tool_calls, raw=item)
            )
        elif text_parts:
            messages.append(Message(role=role if role in {"user", "system"} else "user", content="\n".join(text_parts), raw=item))
    return messages


def messages_to_gemini_contents(messages: List[Message]) -> List[Dict[str, Any]]:
    contents: List[Dict[str, Any]] = []
    for msg in messages:
        if msg.role == "system":
            # Gemini system goes separately; keep as user preamble if needed by caller
            contents.append({"role": "user", "parts": [{"text": f"[System]\n{msg.content}"}]})
            continue
        if msg.role == "tool":
            response_obj: Any
            try:
                response_obj = json.loads(msg.content) if isinstance(msg.content, str) else msg.content
            except Exception:
                response_obj = {"result": msg.content}
            contents.append(
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": msg.name or "tool",
                                "response": response_obj if isinstance(response_obj, dict) else {"result": response_obj},
                            }
                        }
                    ],
                }
            )
            continue
        role = "model" if msg.role == "assistant" else "user"
        parts: List[Dict[str, Any]] = []
        if isinstance(msg.content, str) and msg.content:
            parts.append({"text": msg.content})
        for call in msg.tool_calls:
            parts.append({"functionCall": {"name": call.name, "args": call.arguments or {}}})
        if not parts:
            parts = [{"text": ""}]
        contents.append({"role": role, "parts": parts})
    return contents


def gemini_response(
    *,
    model: str,
    content: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = []
    if content:
        parts.append({"text": content})
    if tool_calls:
        for call in tool_calls:
            parts.append({"functionCall": {"name": call.name, "args": call.arguments or {}}})
    if not parts:
        parts = [{"text": ""}]
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": parts},
                "finishReason": "STOP" if not tool_calls else "STOP",
                "index": 0,
            }
        ],
        "modelVersion": model,
        "usageMetadata": {
            "promptTokenCount": 0,
            "candidatesTokenCount": 0,
            "totalTokenCount": 0,
        },
    }


# ---------- Responses API ----------

def responses_input_to_messages(body: Dict[str, Any]) -> List[Message]:
    messages: List[Message] = []
    instructions = body.get("instructions")
    if instructions:
        messages.append(Message(role="system", content=str(instructions)))

    raw_input = body.get("input")
    if isinstance(raw_input, str):
        messages.append(Message(role="user", content=raw_input))
        return messages

    for item in raw_input or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type")
        role = str(item.get("role") or "")
        if itype == "message" or role:
            role = role or "user"
            content = item.get("content")
            text = ""
            if isinstance(content, str):
                text = content
            elif isinstance(content, list):
                chunks = []
                for c in content:
                    if isinstance(c, dict) and c.get("type") in {"input_text", "output_text", "text"}:
                        chunks.append(str(c.get("text") or ""))
                    elif isinstance(c, str):
                        chunks.append(c)
                text = "\n".join(chunks)
            messages.append(Message(role=role if role in {"system", "user", "assistant"} else "user", content=text, raw=item))
        elif itype == "function_call":
            args = item.get("arguments")
            parsed = {}
            if isinstance(args, dict):
                parsed = args
            elif isinstance(args, str) and args.strip():
                try:
                    value = json.loads(args)
                    parsed = value if isinstance(value, dict) else {"value": value}
                except Exception:
                    parsed = {"_raw": args}
            messages.append(
                Message(
                    role="assistant",
                    content="",
                    tool_calls=[
                        ToolCall(
                            id=str(item.get("call_id") or item.get("id") or call_id()),
                            name=str(item.get("name") or ""),
                            arguments=parsed,
                        )
                    ],
                    raw=item,
                )
            )
        elif itype == "function_call_output":
            messages.append(
                Message(
                    role="tool",
                    content=str(item.get("output") or ""),
                    tool_call_id=str(item.get("call_id") or ""),
                    raw=item,
                )
            )
    return messages


def responses_tools_to_defs(raw: Any) -> List[ToolDef]:
    defs: List[ToolDef] = []
    for item in raw or []:
        if not isinstance(item, dict):
            continue
        if item.get("type") == "function":
            name = str(item.get("name") or "")
            if not name:
                continue
            defs.append(
                ToolDef(
                    name=name,
                    description=str(item.get("description") or ""),
                    parameters=item.get("parameters") if isinstance(item.get("parameters"), dict) else {},
                    raw=item,
                )
            )
        elif item.get("type") == "function" or item.get("name"):
            # already handled / flat
            pass
    # Also accept OpenAI chat style nested
    if not defs:
        from .models.canonical import openai_tools_to_defs

        return openai_tools_to_defs(raw)
    return defs


def build_responses_response(
    *,
    model: str,
    content: str = "",
    tool_calls: Optional[List[ToolCall]] = None,
    usage: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    from app.engine.xyml import responses_tool_items

    output: List[Dict[str, Any]] = []
    if content:
        output.append(
            {
                "id": f"msg_{random_id(12)}",
                "type": "message",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": content}],
            }
        )
    if tool_calls:
        sdk_like = [
            type("C", (), {"id": c.id, "name": c.name, "input": c.arguments})() for c in tool_calls
        ]
        try:
            output.extend(responses_tool_items(sdk_like))
        except Exception:
            for c in tool_calls:
                output.append(
                    {
                        "id": f"fc_{random_id(12)}",
                        "type": "function_call",
                        "status": "completed",
                        "call_id": c.id or call_id(),
                        "name": c.name,
                        "arguments": json.dumps(c.arguments or {}, ensure_ascii=False),
                    }
                )
    status = "completed"
    return {
        "id": f"resp_{random_id(16)}",
        "object": "response",
        "created_at": unix_now(),
        "status": status,
        "model": model,
        "output": output,
        "usage": {
            "input_tokens": (usage or {}).get("prompt_tokens") or 0,
            "output_tokens": (usage or {}).get("completion_tokens") or 0,
            "total_tokens": (usage or {}).get("total_tokens") or 0,
        },
    }
