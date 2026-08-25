"""Stable OpenAI chat.completion streaming (native + prompt FC)."""

from __future__ import annotations

import json
from typing import Any, AsyncIterator, Dict, List, Optional

from ..fc.parse import create_sieve, parse_text_to_calls, to_openai_tool_calls
from ..models.canonical import ToolCall, ToolDef
from ..util.ids import completion_id, unix_now
from ..util.sse import done_frame, format_sse, iter_sse_events


def _chunk(
    *,
    model: str,
    delta: Dict[str, Any],
    finish_reason: Optional[str] = None,
    chunk_id: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": chunk_id or completion_id(),
        "object": "chat.completion.chunk",
        "created": unix_now(),
        "model": model,
        "choices": [
            {
                "index": 0,
                "delta": delta,
                "finish_reason": finish_reason,
                "logprobs": None,
            }
        ],
    }


async def stream_native_passthrough(
    line_iter: AsyncIterator[str],
    *,
    model: str,
) -> AsyncIterator[str]:
    """Forward upstream OpenAI SSE with proper multi-line event framing."""
    finished = False
    try:
        async for event in iter_sse_events(line_iter):
            if event.is_done:
                finished = True
                yield done_frame()
                return
            payload = event.json()
            if payload is None:
                # Pass through non-JSON data frames as-is
                if event.data:
                    yield format_sse(event.data, event=event.event)
                continue
            if model and payload.get("model") != model:
                payload["model"] = model
            yield format_sse(payload, event=event.event)
    except Exception as exc:  # noqa: BLE001
        # Emit a final error-shaped chunk then DONE so clients unblock
        yield format_sse(
            _chunk(
                model=model,
                delta={"content": f"\n[stream error: {exc}]"},
                finish_reason="stop",
            )
        )
        yield done_frame()
        return
    if not finished:
        yield done_frame()


async def stream_prompt_fc(
    line_iter: AsyncIterator[str],
    *,
    model: str,
    tools: List[ToolDef],
    protocol: str,
    strip_think: bool,
) -> AsyncIterator[str]:
    sieve = create_sieve(tools, protocol=protocol)
    chunk_id = completion_id()
    yield format_sse(_chunk(model=model, delta={"role": "assistant"}, chunk_id=chunk_id))

    full_text_parts: List[str] = []
    emitted_calls = False
    pending_calls: List[ToolCall] = []

    async def _emit_calls(calls: List[ToolCall]) -> AsyncIterator[str]:
        nonlocal emitted_calls
        if not calls:
            return
        emitted_calls = True
        openai_calls = to_openai_tool_calls(calls)
        for index, tc in enumerate(openai_calls):
            yield format_sse(
                _chunk(
                    model=model,
                    delta={
                        "tool_calls": [
                            {
                                "index": index,
                                "id": tc.get("id"),
                                "type": "function",
                                "function": {
                                    "name": (tc.get("function") or {}).get("name"),
                                    "arguments": (tc.get("function") or {}).get("arguments", ""),
                                },
                            }
                        ]
                    },
                    chunk_id=chunk_id,
                )
            )
        yield format_sse(
            _chunk(model=model, delta={}, finish_reason="tool_calls", chunk_id=chunk_id)
        )
        yield done_frame()

    def _raw_to_calls(calls_raw: List[Any]) -> List[ToolCall]:
        calls: List[ToolCall] = []
        for item in calls_raw:
            name = getattr(item, "name", None) or (item.get("name") if isinstance(item, dict) else "")
            call_id = getattr(item, "id", None) or (item.get("id") if isinstance(item, dict) else "")
            arguments = getattr(item, "input", None)
            if arguments is None and isinstance(item, dict):
                arguments = item.get("input") or item.get("arguments") or {}
            if not isinstance(arguments, dict):
                arguments = {"value": arguments}
            calls.append(ToolCall(id=str(call_id or ""), name=str(name or ""), arguments=arguments))
        return calls

    try:
        async for event in iter_sse_events(line_iter):
            if event.is_done:
                break
            payload = event.json()
            if not payload:
                continue
            choices = payload.get("choices") or []
            if not choices:
                continue
            delta = (choices[0] or {}).get("delta") or {}
            piece = delta.get("content")
            if not piece:
                message = (choices[0] or {}).get("message") or {}
                piece = message.get("content")
            if not piece:
                continue
            full_text_parts.append(str(piece))
            for sev in sieve.process_chunk(str(piece)):
                if sev.get("type") == "content":
                    text = str(sev.get("text") or "")
                    if text:
                        yield format_sse(
                            _chunk(model=model, delta={"content": text}, chunk_id=chunk_id)
                        )
                elif sev.get("type") == "tool_calls":
                    # 先继续消费上游直到 [DONE]，让 2api 完成 Usage 统计；
                    # 工具调用帧在上游连接关闭后再发给客户端。
                    pending_calls = _raw_to_calls(sev.get("calls") or [])

        for sev in sieve.flush():
            if sev.get("type") == "content":
                text = str(sev.get("text") or "")
                if text:
                    yield format_sse(_chunk(model=model, delta={"content": text}, chunk_id=chunk_id))
            elif sev.get("type") == "tool_calls" and not emitted_calls:
                pending_calls = _raw_to_calls(sev.get("calls") or [])

        if not emitted_calls:
            full_text = "".join(full_text_parts)
            calls = parse_text_to_calls(
                full_text, tools, protocol=protocol, strip_think=strip_think
            )
            if calls:
                pending_calls = calls
        if pending_calls and not emitted_calls:
            async for frame in _emit_calls(pending_calls):
                yield frame
            return
            yield format_sse(_chunk(model=model, delta={}, finish_reason="stop", chunk_id=chunk_id))
        yield done_frame()
    except Exception as exc:  # noqa: BLE001
        yield format_sse(
            _chunk(
                model=model,
                delta={"content": f"\n[stream error: {exc}]"},
                finish_reason="stop",
                chunk_id=chunk_id,
            )
        )
        yield done_frame()
