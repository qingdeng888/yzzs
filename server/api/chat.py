"""
server/api/chat.py - /v1/chat/completions OpenAI 兼容接口
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
import uuid
from typing import Optional, AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.background import BackgroundTask
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from ..auth import require_api_key
from ..db import get_db, session_scope
from ..models import ApiKey, Usage
from ..ratelimit import get_limiter
from ..account_pool import get_pool
from ..config import AppConfig
from ctyun_api import CtyunClient

log = logging.getLogger("api.chat")
router = APIRouter()

# 模型名映射(OpenAI 风格 → 云智 key_model)
MODEL_MAP = {
    "deepseek-v4": "TEXT_DEEPSEEK_V4",
    "glm-5.2": "TEXT_GLM_5.2",
    "qwen-3.7": "TEXT_QWEN_3.7",
    "qwen-3-32b": "TEXT_A13",
    "deepseek": "TEXT_DEEPSEEK_V4",
    "glm": "TEXT_GLM_5.2",
    "qwen": "TEXT_QWEN_3.7",
}


class ChatMessage(BaseModel):
    role: str
    content: str
    name: Optional[str] = None


class ChatRequest(BaseModel):
    model: str = "glm-5.2"
    messages: list[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    # 扩展:用户可在 extra_body 里指定
    enable_thinking: Optional[bool] = None
    web_search: Optional[bool] = None
    # OpenAI 兼容字段(忽略)
    max_tokens: Optional[int] = None
    presence_penalty: Optional[float] = None
    frequency_penalty: Optional[float] = None
    user: Optional[str] = None


def _record_usage(
    db: Session,
    api_key_id: int,
    account_id: Optional[int],
    model: str,
    status: str,
    prompt_tokens: int,
    completion_tokens: int,
    reasoning_tokens: int,
    latency_ms: int,
    stream: bool,
    error_type: str = "",
    error_msg: str = "",
    client_ip: str = "",
):
    u = Usage(
        api_key_id=api_key_id,
        account_id=account_id,
        model=model,
        status=status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=prompt_tokens + completion_tokens + reasoning_tokens,
        latency_ms=latency_ms,
        stream=stream,
        error_type=error_type,
        error_msg=error_msg,
        client_ip=client_ip,
    )
    db.add(u)
    db.commit()


def _create_pending_usage(db: Session, api_key_id: int, account_id: Optional[int],
                          model: str, stream: bool, client_ip: str) -> int:
    u = Usage(api_key_id=api_key_id, account_id=account_id, model=model,
              status="started", stream=stream, client_ip=client_ip)
    db.add(u)
    db.commit()
    return u.id


def _finish_usage(usage_id: int, status: str, prompt_tokens: int,
                  completion_tokens: int, reasoning_tokens: int,
                  latency_ms: int, error_msg: str = "") -> None:
    with session_scope() as db:
        u = db.get(Usage, usage_id)
        if not u:
            return
        u.status = status
        u.prompt_tokens = prompt_tokens
        u.completion_tokens = completion_tokens
        u.reasoning_tokens = reasoning_tokens
        u.total_tokens = prompt_tokens + completion_tokens + reasoning_tokens
        u.latency_ms = latency_ms
        u.error_msg = error_msg[:2000]
        db.commit()


def _finish_interrupted_usage(usage_id: int) -> None:
    """ASGI 取消流式生成器后仍确保 Usage 不停留在 started。"""
    with session_scope() as db:
        u = db.get(Usage, usage_id)
        if u and u.status == "started":
            u.status = "interrupted"
            u.error_msg = "downstream stream closed before completion"
            db.commit()


def _approx_tokens(text: str) -> int:
    """粗略 token 估算(中文 1.5 字/token, 英文 4 字符/token)"""
    if not text:
        return 0
    cn = sum(1 for c in text if '一' <= c <= '鿿')
    other = len(text) - cn
    return int(cn / 1.5 + other / 4) + 1


def _build_ctyun_messages(messages: list[ChatMessage]) -> list[dict]:
    """OpenAI messages → 云智 messages(保留 user 消息, 注入 verify_id)"""
    out = []
    for m in messages:
        d = {"role": m.role, "content": m.content}
        if m.role == "user":
            d["verify_id"] = uuid.uuid4().hex
            d["ref"] = {"type": "file", "file": []}
        out.append(d)
    return out


async def _acquire_with_wait(pool, timeout: float) -> Optional[object]:
    """选号+排队:有活跃账号但槽位满时轮询等待;全部不可用则快速失败"""
    deadline = time.time() + timeout
    while True:
        entry = pool.acquire()
        if entry:
            return entry
        if pool.count_active() == 0:  # 全部 disabled/cooldown,不必等
            return None
        if time.time() >= deadline:
            return None
        await asyncio.sleep(0.2)


async def _stream_response(
    cfg: AppConfig,
    client: CtyunClient,
    key_model: str,
    messages: list[dict],
    enable_thinking: bool,
    web_search: bool,
    api_key_id: int,
    model_name: str,
    client_ip: str,
    entry=None,
    pool=None,
    usage_id: Optional[int] = None,
) -> AsyncGenerator[bytes, None]:
    """生成 SSE 字节流。持有 entry 并发槽位,结束时统一 release + 按需删会话"""
    chat_id = f"chatcmpl-{uuid.uuid4().hex[:24]}"
    created = int(time.time())
    started = time.time()

    # 处理 SSE
    full_content = ""
    full_reasoning = ""
    account_id: Optional[int] = entry.account_id if entry else None
    error_msg = ""
    status = "ok"
    conversation_id: Optional[str] = None
    completed = False
    try:
        # 在线程池中跑同步 requests(在 try 内,异常也能走 finally 释放槽位)
        resp = await asyncio.to_thread(
            client.chat, key_model, messages,
            enable_thinking=enable_thinking,
            web_search=web_search,
            stream=True,
        )
        for raw_line in resp.iter_lines():
            if not raw_line:
                continue
            line = raw_line.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if not data:
                continue
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue

            # 首 chunk 携带 conversation_id(该 chunk 无 choices,会被下面跳过)
            if conversation_id is None:
                conversation_id = chunk.get("conversation_id")

            choices = chunk.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content", "")
            reasoning = delta.get("reasoning_content", "")

            # mojibake 修复
            if content:
                try:
                    content = content.encode("latin-1").decode("utf-8")
                except Exception:
                    pass
                full_content += content
            if reasoning:
                try:
                    reasoning = reasoning.encode("latin-1").decode("utf-8")
                except Exception:
                    pass
                full_reasoning += reasoning

            out_delta = {"role": "assistant", "content": content}
            if enable_thinking and reasoning:
                out_delta["reasoning_content"] = reasoning

            finish = choices[0].get("finish_reason")
            out_chunk = {
                "id": chat_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": out_delta,
                    "finish_reason": finish,
                }],
            }
            yield f"data: {json.dumps(out_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
            if finish:
                # 再发一个 [DONE]
                yield b"data: [DONE]\n\n"
                completed = True
                break
        else:
            # 流自然结束(没有 finish_reason)
            yield b"data: [DONE]\n\n"
            completed = True
    except GeneratorExit:
        # 客户端关闭连接，仍然落一条中断请求统计。
        status = "interrupted"
    except asyncio.CancelledError:
        # ToolForge 在识别到 Prompt-FC 后会主动结束上游 SSE；不要让取消
        # 信号跳过 finally 中的 Usage 写入。
        status = "interrupted"
        log.info("stream cancelled by downstream, recording partial usage")
    except Exception as e:
        log.exception("stream error")
        error_msg = str(e)
        status = "fail"
        err_chunk = {
            "id": chat_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [{
                "index": 0,
                "delta": {"content": "\n[upstream stream error]"},
                "finish_reason": "stop",
            }],
        }
        yield f"data: {json.dumps(err_chunk, ensure_ascii=False)}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"
    finally:
        latency = int((time.time() - started) * 1000)
        prompt_tokens = sum(_approx_tokens(m["content"]) for m in messages)
        completion_tokens = _approx_tokens(full_content)
        reasoning_tokens = _approx_tokens(full_reasoning)
        # 释放并发槽位(无论正常/取消/异常都执行)
        try:
            pool.release(entry)
        except Exception:
            pass
        # 仅当完整回复结束后按配置删除本会话(客户端断开时 completed=False,不删)
        if completed and conversation_id and entry and entry.delete_conversation_after:
            try:
                j = await asyncio.to_thread(client.remove_conversation, conversation_id)
                log.info(f"账号 {entry.account_id} 已删除会话 {conversation_id}: {j.get('resultMsg')}")
            except Exception as e:
                log.warning(f"删除会话 {conversation_id} 失败: {e}")
        if usage_id:
            # 用线程执行，避免客户端取消信号打断 SQLite 提交。
            await asyncio.shield(asyncio.to_thread(
                _finish_usage, usage_id, status, prompt_tokens,
                completion_tokens, reasoning_tokens, latency, error_msg,
            ))
        else:
            with session_scope() as db:
                _record_usage(db, api_key_id, account_id, model_name, status,
                              prompt_tokens, completion_tokens, reasoning_tokens,
                              latency, stream=True, error_msg=error_msg, client_ip=client_ip)


@router.post("/v1/chat/completions")
async def chat_completions(
    body: ChatRequest,
    request: Request,
    api_key: ApiKey = Depends(require_api_key),
):
    # 限流
    get_limiter().check(api_key)
    cfg: AppConfig = request.app.state.config

    # 解析模型(-nothing 后缀 = 关闭深度思考的变种)
    model = (body.model or "").lower()
    thinking_forced_off = False
    if model.endswith("-nothing"):
        thinking_forced_off = True
        base_model = model[: -len("-nothing")]
    else:
        base_model = model
    key_model = MODEL_MAP.get(base_model, body.model)

    # 构造 ctyun messages
    ctyun_messages = _build_ctyun_messages(body.messages)
    if not ctyun_messages:
        raise HTTPException(
            status_code=400,
            detail={"error": {"message": "messages is empty", "type": "invalid_request", "code": "empty_messages"}},
        )

    if thinking_forced_off:
        enable_thinking = False                          # -nothing 变种:强制关闭
    elif body.enable_thinking is not None:
        enable_thinking = bool(body.enable_thinking)     # extra_body 显式覆盖
    else:
        enable_thinking = True                           # 默认开启深度思考
    web_search = body.web_search if body.web_search is not None else cfg.api.enable_web_search_default

    client_ip = request.client.host if request.client else ""
    pool = get_pool()

    # 选账号并获取/登录 client(并发槽位满时排队等待,超时 503)
    max_attempts = max(1, cfg.api.retry_max + 1)
    last_err: Optional[Exception] = None
    for attempt in range(max_attempts):
        entry = await _acquire_with_wait(pool, cfg.api.acquire_wait_timeout_seconds)
        if not entry:
            raise HTTPException(
                status_code=503,
                detail={"error": {"message": "No available account (all disabled/in cooldown/saturated)", "type": "service_unavailable", "code": "no_account"}},
            )
        # 阶段1: 登录(可能失败 -> 释放槽位后重试下一个账号)
        try:
            client = await asyncio.to_thread(pool.try_get_or_login, entry.account_id)
        except Exception as e:
            last_err = e
            pool.release(entry)
            log.warning(f"账号 {entry.account_id} 登录失败: {e}")
            continue
        if client is None:
            last_err = RuntimeError("client is None after login")
            pool.release(entry)
            continue

        # 阶段2: 分发
        if body.stream:
            with session_scope() as usage_db:
                usage_id = _create_pending_usage(
                    usage_db, api_key.id, entry.account_id, body.model,
                    stream=True, client_ip=client_ip,
                )
            # 槽位所有权移交给生成器,由 _stream_response 的 finally 释放
            return StreamingResponse(
                _stream_response(
                    cfg, client, key_model, ctyun_messages,
                    enable_thinking, web_search,
                    api_key.id, body.model, client_ip,
                    entry=entry, pool=pool,
                    usage_id=usage_id,
                ),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                },
                background=BackgroundTask(_finish_interrupted_usage, usage_id),
            )
        else:
            # 非流式: 调同步流,自己拼
            started = time.time()
            try:
                conv_holder: dict = {}
                resp_iter = client._openai_stream(
                    key_model, body.model, ctyun_messages, enable_thinking,
                    on_conversation_id=lambda cid: conv_holder.setdefault('id', cid),
                )
                content = ""
                reasoning = ""
                for c in resp_iter:
                    d = c.get("choices", [{}])[0].get("delta", {})
                    content += d.get("content", "")
                    reasoning += d.get("reasoning_content", "") or ""
                latency = int((time.time() - started) * 1000)
                prompt_tokens = sum(_approx_tokens(m["content"]) for m in ctyun_messages)
                completion_tokens = _approx_tokens(content)
                reasoning_tokens = _approx_tokens(reasoning)
                with session_scope() as db:
                    _record_usage(
                        db, api_key.id, entry.account_id, body.model, "ok",
                        prompt_tokens, completion_tokens, reasoning_tokens,
                        latency, stream=False, client_ip=client_ip,
                    )
                pool.mark_success(entry.account_id)
                # 配置了"完成后删会话"则删除本请求创建的会话
                conv_id = conv_holder.get('id')
                if entry.delete_conversation_after and conv_id:
                    try:
                        j = await asyncio.to_thread(client.remove_conversation, conv_id)
                        log.info(f"账号 {entry.account_id} 已删除会话 {conv_id}: {j.get('resultMsg')}")
                    except Exception as e:
                        log.warning(f"删除会话 {conv_id} 失败: {e}")
                return JSONResponse({
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": body.model,
                    "choices": [{
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": content,
                            "reasoning_content": reasoning if enable_thinking else None,
                        },
                        "finish_reason": "stop",
                    }],
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens,
                    },
                })
            except Exception as e:
                log.warning(f"非流式 chat 失败 (attempt {attempt+1}): {e}")
                last_err = e
                pool.mark_error(entry.account_id, str(e))
                continue
            finally:
                pool.release(entry)

    # 全部重试失败
    log.error(f"所有账号都失败: {last_err}")
    raise HTTPException(
        status_code=502,
        detail={"error": {
            "message": "All upstream accounts failed",
            "type": "upstream_error",
            "code": "all_accounts_failed",
        }},
    )


async def _async_iter_sync(sync_iter):
    """把同步 generator 包成 async generator"""
    for item in sync_iter:
        yield item
