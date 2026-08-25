"""
server/auth.py - 后台 cookie 鉴权 + API Key 鉴权
"""
from __future__ import annotations
import hashlib
import hmac
import json
import secrets
import time
from datetime import datetime, timezone
from typing import Optional
import bcrypt
from fastapi import Depends, HTTPException, Request
from sqlalchemy.orm import Session

from .db import get_db
from .models import AdminUser, ApiKey
from .config import AppConfig


def hash_password(plain: str) -> str:
    """bcrypt 哈希(直接调用 bcrypt 包,避免 passlib 与 bcrypt>=4.1 不兼容)"""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False


def hash_api_key(key: str) -> str:
    """API Key 存 SHA256 hash(类似 OpenAI)"""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


def make_api_key() -> str:
    """生成新的 API Key: sk-xxx (32 字节 base62)"""
    raw = secrets.token_urlsafe(32)
    return f"sk-{raw}"


# ============== 后台 Cookie ==============

ADMIN_COOKIE = "ctyun_admin"


def make_admin_token(username: str, secret: str, expire_seconds: int) -> str:
    """简单的 HMAC 签名 cookie"""
    payload = json.dumps({"u": username, "exp": int(time.time()) + expire_seconds}, separators=(",", ":"))
    sig = hmac.new(secret.encode("utf-8"), payload.encode("utf-8"), hashlib.sha256).hexdigest()
    import base64
    token = base64.urlsafe_b64encode(f"{payload}|{sig}".encode("utf-8")).decode("ascii")
    return token


def verify_admin_token(token: str, secret: str) -> Optional[dict]:
    import base64
    try:
        raw = base64.urlsafe_b64decode(token.encode("ascii")).decode("utf-8")
        payload_str, sig = raw.rsplit("|", 1)
        expected = hmac.new(secret.encode("utf-8"), payload_str.encode("utf-8"), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected):
            return None
        data = json.loads(payload_str)
        if data.get("exp", 0) < time.time():
            return None
        return data
    except Exception:
        return None


def get_current_admin(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[AdminUser]:
    """返回 AdminUser 或 None。

    注意:直接从 request.cookies 读取,这样无论作为 FastAPI 依赖被注入,
    还是在路由内被直接调用,都能正确拿到 cookie(直接调用时 Cookie(...)
    参数不会被 DI 填充)。
    """
    cfg: AppConfig = request.app.state.config
    ctyun_admin = request.cookies.get(ADMIN_COOKIE)
    if not ctyun_admin:
        return None
    payload = verify_admin_token(ctyun_admin, cfg.admin.session_secret or cfg.get_session_secret())
    if not payload:
        return None
    u = db.query(AdminUser).filter(AdminUser.username == payload["u"]).first()
    return u


def require_admin(admin: Optional[AdminUser] = Depends(get_current_admin)) -> AdminUser:
    """依赖:必须已登录后台,否则 302 跳登录页(API 接口用 401)"""
    if admin is None:
        raise HTTPException(status_code=401, detail="admin auth required")
    return admin


# ============== API Key 鉴权 ==============


def get_api_key_from_request(request: Request) -> Optional[str]:
    """从 Authorization header 提取 API Key"""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if not auth:
        return None
    parts = auth.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip()


def require_api_key(
    request: Request,
    db: Session = Depends(get_db),
) -> ApiKey:
    """FastAPI 依赖:校验 Bearer Token,返回 ApiKey 或 401"""
    token = get_api_key_from_request(request)
    if not token:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Missing or invalid Authorization header", "type": "auth", "code": "invalid_api_key"}},
        )
    key_hash = hash_api_key(token)
    api_key = db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first()
    if not api_key:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "Invalid API key", "type": "auth", "code": "invalid_api_key"}},
        )
    if api_key.status != "active":
        raise HTTPException(
            status_code=403,
            detail={"error": {"message": f"API key {api_key.name} is {api_key.status}", "type": "auth", "code": "key_disabled"}},
        )
    if api_key.expires_at and api_key.expires_at < datetime.utcnow():
        raise HTTPException(
            status_code=403,
            detail={"error": {"message": "API key expired", "type": "auth", "code": "key_expired"}},
        )
    # 更新最后使用时间
    api_key.last_used_at = datetime.utcnow()
    db.commit()
    return api_key
