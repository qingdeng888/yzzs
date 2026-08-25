"""
server/models.py - ORM 模型
"""
from __future__ import annotations
from datetime import datetime
from sqlalchemy import String, Integer, BigInteger, DateTime, Text, Boolean, ForeignKey, Index
from sqlalchemy.orm import Mapped, mapped_column, relationship
from .db import Base


def utcnow() -> datetime:
    return datetime.utcnow()


class AdminUser(Base):
    """后台管理员(单用户)"""
    __tablename__ = "admin_users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Account(Base):
    """云智账号"""
    __tablename__ = "accounts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_enc: Mapped[str] = mapped_column(String(255))  # bcrypt 加密
    tenant_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    user_id: Mapped[int | None] = mapped_column(BigInteger, nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="active")  # active / disabled / locked
    remark: Mapped[str] = mapped_column(String(255), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_error: Mapped[str] = mapped_column(Text, default="")
    cooldown_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)  # 短期封禁到期时间
    error_count: Mapped[int] = mapped_column(Integer, default=0)
    concurrency_limit: Mapped[int] = mapped_column(Integer, default=2, server_default="2", nullable=False)  # 并发上限
    delete_conversation_after: Mapped[bool] = mapped_column(Boolean, default=False, server_default="0", nullable=False)  # 对话完成后删会话

    session: Mapped["CachedSession | None"] = relationship(
        "CachedSession", uselist=False, back_populates="account", cascade="all, delete-orphan"
    )

    @property
    def has_session(self) -> bool:
        """是否有缓存 session(仅判断存在,不判断是否过期;过期由 account_pool 自动重登)"""
        return self.session is not None and bool(self.session.yl_auth)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "account": self.account,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "status": self.status,
            "remark": self.remark,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
            "last_error": self.last_error[:200],
            "cooldown_until": self.cooldown_until.isoformat() if self.cooldown_until else None,
            "error_count": self.error_count,
            "has_session": self.session is not None and bool(self.session.yl_auth),
            "concurrency_limit": self.concurrency_limit,
            "delete_conversation_after": self.delete_conversation_after,
        }


class CachedSession(Base):
    """账号的登录 session 缓存(yl-auth + sk)"""
    __tablename__ = "cached_sessions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    account_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="CASCADE"), unique=True, index=True
    )
    yl_auth: Mapped[str] = mapped_column(Text)
    sk: Mapped[str] = mapped_column(Text)
    xuid: Mapped[str] = mapped_column(String(64), default="")
    expires_at: Mapped[datetime] = mapped_column(DateTime)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    account: Mapped[Account] = relationship("Account", back_populates="session")


class ApiKey(Base):
    """API Key(客户端用)"""
    __tablename__ = "api_keys"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64))
    # 存储 hash(明文只在创建时返回一次)
    key_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    # 用于显示的前4后4位
    key_prefix: Mapped[str] = mapped_column(String(16))
    key_suffix: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16), default="active")  # active / disabled
    daily_quota: Mapped[int] = mapped_column(Integer, default=0)  # 0=无限
    monthly_quota: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    remark: Mapped[str] = mapped_column(String(255), default="")

    def to_dict(self, include_plain: str = None) -> dict:
        d = {
            "id": self.id,
            "name": self.name,
            "key": f"{self.key_prefix}***{self.key_suffix}",
            "status": self.status,
            "daily_quota": self.daily_quota,
            "monthly_quota": self.monthly_quota,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_used_at": self.last_used_at.isoformat() if self.last_used_at else None,
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "remark": self.remark,
        }
        if include_plain:
            d["key_plain"] = include_plain
        return d


class Usage(Base):
    """每次请求的统计记录"""
    __tablename__ = "usage"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    api_key_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("api_keys.id", ondelete="SET NULL"), nullable=True, index=True
    )
    account_id: Mapped[int | None] = mapped_column(
        Integer, ForeignKey("accounts.id", ondelete="SET NULL"), nullable=True, index=True
    )
    model: Mapped[str] = mapped_column(String(64), default="", index=True)
    status: Mapped[str] = mapped_column(String(16), default="ok")  # ok / fail
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    reasoning_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    stream: Mapped[bool] = mapped_column(Boolean, default=False)
    error_type: Mapped[str] = mapped_column(String(64), default="")
    error_msg: Mapped[str] = mapped_column(Text, default="")
    client_ip: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "api_key_id": self.api_key_id,
            "account_id": self.account_id,
            "model": self.model,
            "status": self.status,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
            "stream": self.stream,
            "error_type": self.error_type,
            "error_msg": self.error_msg[:200],
            "client_ip": self.client_ip,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


# 索引
Index("ix_usage_status_created", Usage.status, Usage.created_at)
Index("ix_usage_apikey_created", Usage.api_key_id, Usage.created_at)
Index("ix_usage_account_created", Usage.account_id, Usage.created_at)
