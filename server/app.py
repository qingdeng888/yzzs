"""
server/app.py - FastAPI 应用工厂
"""
from __future__ import annotations
import asyncio
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from . import crypto
from .config import AppConfig, load_config
from .db import init_engine, create_all
from .ratelimit import init_limiter
from .account_pool import init_pool, get_pool
from .models import AdminUser, ApiKey
from . import session_store
from .auth import hash_password, verify_password, hash_api_key
from .db import session_scope

from .api import chat as chat_api
from .api import models as models_api
from .api import health as health_api
from .web import routes as web_routes

log = logging.getLogger("app")

STATIC_DIR = Path(__file__).parent / "web" / "static"


def _bootstrap_admin(cfg: AppConfig):
    """确保有一个 admin 用户(从 config.yaml 读初始密码)"""
    with session_scope() as db:
        u = db.query(AdminUser).filter(AdminUser.username == cfg.admin.username).first()
        if not u:
            if not cfg.admin.password or cfg.admin.password == "changeme":
                raise RuntimeError("首次启动必须设置 CTYUN_ADMIN_PASSWORD，禁止使用默认管理员密码")
            u = AdminUser(
                username=cfg.admin.username,
                password_hash=hash_password(cfg.admin.password),
            )
            db.add(u)
            db.commit()
            log.info(f"创建初始 admin 用户: {cfg.admin.username}")
        elif not u.password_hash:
            u.password_hash = hash_password(cfg.admin.password)
            db.commit()
        elif verify_password("changeme", u.password_hash):
            if not cfg.admin.password or cfg.admin.password == "changeme":
                raise RuntimeError("检测到默认管理员密码，请设置 CTYUN_ADMIN_PASSWORD 后重启")
            u.password_hash = hash_password(cfg.admin.password)
            db.commit()


def _bootstrap_internal_api_key():
    """为 Compose 内部 ToolForge 幂等创建 API Key，不在页面显示明文。"""
    plain = os.environ.get("CTYUN_INTERNAL_API_KEY", "").strip()
    if not plain:
        return
    with session_scope() as db:
        key_hash = hash_api_key(plain)
        if db.query(ApiKey).filter(ApiKey.key_hash == key_hash).first():
            return
        db.add(ApiKey(
            name="toolforge-internal",
            key_hash=key_hash,
            key_prefix=plain[:7],
            key_suffix=plain[-4:],
            remark="Compose ToolForge 内部访问",
        ))
        db.commit()
def _bootstrap_account_passwords(cfg: AppConfig, pool):
    """启动时把所有账号的明文密码注入到内存池(从 DB 解密)"""
    with session_scope() as db:
        for a in session_store.list_accounts(db):
            try:
                plain = crypto.decrypt(a.password_enc)
                pool.set_password(a.id, plain)
            except Exception as e:
                log.warning(f"账号 {a.id} 密码解密失败: {e}")


def _warmup_thread(pool):
    """后台线程预热所有账号"""
    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(pool.warmup_all())
        finally:
            loop.close()
    threading.Thread(target=run, daemon=True, name="warmup").start()


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # 启动
        log.info("初始化数据库...")
        init_engine(cfg.database.url, echo=cfg.database.echo)
        create_all()

        log.info("初始化加密模块...")
        crypto.init_crypto(cfg.get_session_secret())

        log.info("初始化限流器...")
        init_limiter(cfg)

        log.info("初始化账号池...")
        pool = init_pool(cfg)
        _bootstrap_account_passwords(cfg, pool)
        _warmup_thread(pool)

        log.info("初始化 admin 用户...")
        _bootstrap_admin(cfg)
        _bootstrap_internal_api_key()

        log.info("启动完成")
        yield

        # 关闭
        log.info("关闭中...")

    app = FastAPI(
        title="云智助手 2api",
        description="把云智助手网页端对话转成 OpenAI 兼容 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    app.state.config = cfg

    @app.middleware("http")
    async def admin_csrf_middleware(request: Request, call_next):
        if request.method == "POST" and request.url.path.startswith("/admin"):
            if not await web_routes.check_csrf(request):
                return JSONResponse(status_code=403, content={"detail": "CSRF validation failed"})
        return await call_next(request)

    # 路由
    app.include_router(health_api.router)
    app.include_router(models_api.router)
    app.include_router(chat_api.router)
    app.include_router(web_routes.router)

    if STATIC_DIR.exists():
        app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

    # 全局错误处理
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        log.exception("unhandled error")
        return JSONResponse(
            status_code=500,
            content={"error": {"message": "Internal server error", "type": "internal_error", "code": "internal"}},
        )

    @app.get("/")
    async def index():
        return {
            "name": "云智助手 2api",
            "version": "1.0.0",
            "admin": "/admin/",
            "docs": "/docs",
            "openai_api": "/v1/chat/completions",
        }

    return app
