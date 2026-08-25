"""
server/account_pool.py - 多账号轮询池

职责:
- 内存中维护 active 账号列表
- round-robin 分发账号
- 维护冷却期(短期封禁,默认 60s)
- 后台异步刷新过期 session
- 健康检查与自动重登
"""
from __future__ import annotations
import asyncio
import logging
import threading
import time
from datetime import datetime, timezone, timedelta
from typing import Optional
from sqlalchemy import select
from sqlalchemy.orm import Session

from .db import session_scope
from .models import Account, CachedSession
from . import session_store
from ctyun_api import CtyunClient
from .config import AppConfig

log = logging.getLogger("account_pool")


class AccountEntry:
    """单个账号的运行时状态(内存)"""

    def __init__(self, account: Account):
        self.account_id = account.id
        self.account_name = account.account
        self.password = ""  # 由调用方填充(从 DB 读出后解密)
        self.client: Optional[CtyunClient] = None
        self.cooldown_until: Optional[float] = None  # epoch seconds
        self.locked = False
        self.last_error: str = ""
        self.error_count = 0
        # 并发槽位(由 acquire/release 在池锁内维护)
        self.active_requests: int = 0
        self.concurrency_limit: int = 2
        self.delete_conversation_after: bool = False
        # 每个账号一把登录锁:串行网络登录,防并发重复登录(IAM 40050)
        self._login_lock = threading.Lock()

    def is_available(self, now: float) -> bool:
        if self.locked:
            return False
        if self.cooldown_until and now < self.cooldown_until:
            return False
        return True

    def has_capacity(self) -> bool:
        return self.active_requests < self.concurrency_limit


class AccountPool:
    def __init__(self, cfg: AppConfig):
        self.cfg = cfg
        self._entries: dict[int, AccountEntry] = {}
        self._lock = threading.RLock()
        self._rr_counter = 0
        self._last_refresh = 0.0

    # ---- 加载/同步 ----

    def reload(self):
        """从 DB 重新加载账号列表(用于后台增删账号后)"""
        with self._lock, session_scope() as db:
            accounts = list(db.scalars(select(Account)))
            current_ids = {a.id for a in accounts}
            # 移除不存在的
            for aid in list(self._entries.keys()):
                if aid not in current_ids:
                    del self._entries[aid]
            # 添加新账号
            for a in accounts:
                if a.id not in self._entries:
                    self._entries[a.id] = AccountEntry(a)
                e = self._entries[a.id]
                e.account_name = a.account
                e.password = ""  # 等真正要用时再查
                e.concurrency_limit = max(1, a.concurrency_limit or 2)
                e.delete_conversation_after = bool(a.delete_conversation_after)
                if a.status == "disabled":
                    e.locked = True
                else:
                    e.locked = False
        log.info(f"账号池已重载,当前 {len(self._entries)} 个账号")

    def get_password(self, account_id: int) -> Optional[str]:
        """从 DB 读密码(明文,因为存储是 bcrypt,这里需要解)"""
        with session_scope() as db:
            a = db.get(Account, account_id)
            if not a:
                return None
            # password_enc 是 bcrypt hash,不能反解!
            # 所以登录用的明文密码必须另外存 / 缓存
            # 这里用 _plain_cache
            return self._entries[account_id].password if account_id in self._entries else None

    def set_password(self, account_id: int, password: str):
        """创建账号时由调用方设置明文密码(也存到内存)"""
        with self._lock:
            if account_id in self._entries:
                self._entries[account_id].password = password

    # ---- 选号 ----

    def acquire(self) -> Optional[AccountEntry]:
        """获取一个可用账号(round-robin),成功即占一个并发槽位"""
        with self._lock:
            now = time.time()
            available = [e for e in self._entries.values()
                         if e.is_available(now) and e.has_capacity()]
            if not available:
                return None
            self._rr_counter = (self._rr_counter + 1) % len(available)
            e = available[self._rr_counter]
            e.active_requests += 1
            return e

    def release(self, entry: AccountEntry):
        """请求结束释放并发槽位(幂等)"""
        with self._lock:
            if entry.active_requests > 0:
                entry.active_requests -= 1

    def count_active(self) -> int:
        """可用账号数(非 disabled/cooldown),用于区分'全不可用'(快速 503)与'已饱和'(排队等待)"""
        with self._lock:
            now = time.time()
            return sum(1 for e in self._entries.values() if e.is_available(now))

    def mark_error(self, account_id: int, err: str, cooldown_seconds: Optional[int] = None):
        """标记账号出错(短期封禁)"""
        seconds = cooldown_seconds or self.cfg.api.retry_cooldown_seconds
        with self._lock:
            if account_id in self._entries:
                e = self._entries[account_id]
                e.cooldown_until = time.time() + seconds
                e.last_error = err[:500]
                e.error_count += 1
        # 同步写 DB
        with session_scope() as db:
            session_store.set_account_cooldown(db, account_id, seconds, err)

    def mark_success(self, account_id: int):
        """标记成功(清错)"""
        with self._lock:
            if account_id in self._entries:
                e = self._entries[account_id]
                e.cooldown_until = None
                e.last_error = ""
                e.error_count = 0
        with session_scope() as db:
            session_store.clear_account_cooldown(db, account_id)

    def lock_account(self, account_id: int, reason: str = ""):
        with self._lock:
            if account_id in self._entries:
                self._entries[account_id].locked = True
        with session_scope() as db:
            session_store.update_account_status(db, account_id, "disabled")
        log.warning(f"账号 {account_id} 已禁用: {reason}")

    # ---- 登录 / 刷新 ----

    def get_client(self, account_id: int) -> Optional[CtyunClient]:
        """拿或构造 CtyunClient(优先用缓存的 session)"""
        with self._lock:
            e = self._entries.get(account_id)
            if not e or not e.password:
                return None
            if e.client and e.client.logged_in and not e.client.is_expired(
                skew_seconds=self.cfg.api.session_refresh_skew_seconds
            ):
                return e.client
        # 需要登录或刷新
        return None

    def do_login(self, account_id: int) -> CtyunClient:
        """执行登录(同步,会阻塞几秒)。账号级锁串行,锁内二次检查防并发重复登录"""
        with self._lock:
            e = self._entries.get(account_id)
            if not e or not e.password:
                raise RuntimeError(f"账号 {account_id} 不存在或密码未设置")
            password = e.password
            account_name = e.account_name
            entry_lock = e._login_lock

        with entry_lock:
            # 另一线程可能刚登录完:拿到锁后再查一次缓存,有效则直接复用
            c = self.get_client(account_id)
            if c:
                return c

            log.info(f"开始登录账号 {account_id} ({account_name})")
            client = CtyunClient(account_name, password)
            try:
                client.login()
            except Exception as ex:
                err = str(ex)
                log.warning(f"登录失败 {account_id}: {err}")
                # 特殊错误: IAM 40050 = 短时间重复登录
                if "40050" in err and self.cfg.api.auto_disable_on_iam_40050:
                    self.lock_account(account_id, f"IAM 40050: {err[:100]}")
                else:
                    self.mark_error(account_id, err)
                raise

            # 登录成功
            with session_scope() as db:
                session_store.set_account_login_success(db, account_id, client.user_id, client.tenant_id)
                session_store.save_session(
                    db, account_id,
                    client.yl_auth, client.sk, client.xuid, client.expires_at,
                )
            with self._lock:
                if account_id in self._entries:
                    self._entries[account_id].client = client
            self.mark_success(account_id)
            log.info(f"账号 {account_id} 登录成功, expires_at={client.expires_at}")
            return client

    def try_get_or_login(self, account_id: int) -> Optional[CtyunClient]:
        """优先用缓存,失败/过期时自动登录"""
        c = self.get_client(account_id)
        if c:
            return c
        try:
            return self.do_login(account_id)
        except Exception:
            return None

    def update_account_settings(self, account_id: int,
                                concurrency_limit: Optional[int] = None,
                                delete_conversation_after: Optional[bool] = None):
        """Web 界面改设置后原地更新内存 entry(不能调 reload(),那会清空全部密码缓存)"""
        with self._lock:
            e = self._entries.get(account_id)
            if not e:
                return
            if concurrency_limit is not None:
                e.concurrency_limit = max(1, int(concurrency_limit))
            if delete_conversation_after is not None:
                e.delete_conversation_after = bool(delete_conversation_after)

    def clear_conversations(self, account_id: int) -> str:
        """清空某账号所有会话(Web 按钮用)。返回结果消息"""
        with self._lock:
            e = self._entries.get(account_id)
            if not e:
                return "账号不存在"
        try:
            client = self.try_get_or_login(account_id)
            if client is None:
                return "获取 client 失败(登录失败)"
            j = client.clear_all_conversations()
            msg = j.get("resultMsg") or "unknown"
            log.info(f"账号 {account_id} 清空会话: resultCode={j.get('resultCode')} resultMsg={msg}")
            return f"已清空会话 (resultMsg={msg})"
        except Exception as ex:
            log.warning(f"账号 {account_id} 清空会话失败: {ex}")
            return f"清空失败: {ex}"

    def ensure_client(self) -> Optional[CtyunClient]:
        """对外 API: 获取一个能用的 client(占用一个槽位,调用方需负责 release)"""
        e = self.acquire()
        if not e:
            return None
        c = self.try_get_or_login(e.account_id)
        if c is None:
            self.release(e)
            return None
        return c

    # ---- 启动时预热 ----

    async def warmup_all(self):
        """启动后异步预热所有 active 账号"""
        await asyncio.sleep(0.5)
        for aid in list(self._entries.keys()):
            try:
                await asyncio.to_thread(self.do_login, aid)
            except Exception as e:
                log.warning(f"预热账号 {aid} 失败: {e}")


# 全局单例
_pool: Optional[AccountPool] = None


def init_pool(cfg: AppConfig) -> AccountPool:
    global _pool
    _pool = AccountPool(cfg)
    _pool.reload()
    return _pool


def get_pool() -> AccountPool:
    if _pool is None:
        raise RuntimeError("AccountPool not initialized")
    return _pool
