"""
server/session_store.py - Account ↔ CachedSession 的 CRUD 封装
"""
from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from .models import Account, CachedSession

log = logging.getLogger("session_store")


def get_account(db: Session, account_id: int) -> Optional[Account]:
    return db.get(Account, account_id)


def get_account_by_name(db: Session, account: str) -> Optional[Account]:
    return db.scalar(select(Account).where(Account.account == account))


def list_accounts(db: Session) -> list[Account]:
    return list(db.scalars(select(Account).order_by(Account.id)))


def create_account(db: Session, account: str, password_enc: str, remark: str = "",
                   concurrency_limit: int = 2,
                   delete_conversation_after: bool = False) -> Account:
    """创建账号(password_enc 应是 server.crypto.encrypt() 后的密文)"""
    a = Account(
        account=account, password_enc=password_enc, remark=remark,
        concurrency_limit=max(1, concurrency_limit),
        delete_conversation_after=bool(delete_conversation_after),
    )
    db.add(a)
    db.flush()
    return a


def update_account_settings(db: Session, account_id: int,
                            concurrency_limit: Optional[int] = None,
                            delete_conversation_after: Optional[bool] = None) -> bool:
    """更新账号的并发/删会话设置"""
    a = db.get(Account, account_id)
    if not a:
        return False
    if concurrency_limit is not None:
        a.concurrency_limit = max(1, concurrency_limit)
    if delete_conversation_after is not None:
        a.delete_conversation_after = bool(delete_conversation_after)
    a.updated_at = datetime.utcnow()
    return True


def delete_account(db: Session, account_id: int) -> bool:
    a = db.get(Account, account_id)
    if not a:
        return False
    db.delete(a)
    return True


def update_account_status(db: Session, account_id: int, status: str) -> None:
    a = db.get(Account, account_id)
    if a:
        a.status = status
        a.updated_at = datetime.utcnow()


def set_account_cooldown(db: Session, account_id: int, seconds: int, error_msg: str = "") -> None:
    a = db.get(Account, account_id)
    if not a:
        return
    from datetime import timedelta
    a.cooldown_until = datetime.utcnow() + timedelta(seconds=seconds)
    a.last_error = error_msg[:1000]
    a.error_count = (a.error_count or 0) + 1
    a.updated_at = datetime.utcnow()


def clear_account_cooldown(db: Session, account_id: int) -> None:
    a = db.get(Account, account_id)
    if not a:
        return
    a.cooldown_until = None
    a.last_error = ""
    a.error_count = 0
    a.updated_at = datetime.utcnow()


def set_account_login_success(db: Session, account_id: int, user_id: int, tenant_id: int) -> None:
    a = db.get(Account, account_id)
    if a:
        a.user_id = user_id
        a.tenant_id = tenant_id
        a.last_login_at = datetime.utcnow()
        a.cooldown_until = None
        a.error_count = 0
        a.last_error = ""
        a.status = "active"
        a.updated_at = datetime.utcnow()


def get_session(db: Session, account_id: int) -> Optional[CachedSession]:
    return db.scalar(select(CachedSession).where(CachedSession.account_id == account_id))


def save_session(db: Session, account_id: int, yl_auth: str, sk: str,
                 xuid: str, expires_at: Optional[datetime]) -> CachedSession:
    s = get_session(db, account_id)
    if s is None:
        s = CachedSession(account_id=account_id)
        db.add(s)
    s.yl_auth = yl_auth
    s.sk = sk
    s.xuid = xuid
    if expires_at is None:
        expires_at = datetime.utcnow() + timedelta(hours=1)
    elif expires_at.tzinfo is not None:
        # 统一存 naive UTC
        expires_at = expires_at.astimezone(timezone.utc).replace(tzinfo=None)
    s.expires_at = expires_at
    s.updated_at = datetime.utcnow()
    db.flush()
    return s


def delete_session(db: Session, account_id: int) -> None:
    s = get_session(db, account_id)
    if s:
        db.delete(s)
