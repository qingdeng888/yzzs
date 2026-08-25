"""
server/db.py - SQLAlchemy 引擎、会话工厂
"""
from __future__ import annotations
import logging
from contextlib import contextmanager
from typing import Iterator
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session, DeclarativeBase

log = logging.getLogger(__name__)


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def init_engine(db_url: str, echo: bool = False):
    """初始化全局引擎(单例)"""
    global _engine, _SessionLocal
    if _engine is not None:
        return _engine

    connect_args = {}
    if db_url.startswith("sqlite"):
        connect_args = {"check_same_thread": False, "timeout": 30}

    _engine = create_engine(
        db_url,
        echo=echo,
        connect_args=connect_args,
        pool_pre_ping=True,
    )
    _SessionLocal = sessionmaker(
        bind=_engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
    log.info(f"DB engine initialized: {db_url}")
    return _engine


def get_engine():
    if _engine is None:
        raise RuntimeError("DB engine not initialized, call init_engine() first")
    return _engine


def get_session_factory():
    if _SessionLocal is None:
        raise RuntimeError("DB not initialized")
    return _SessionLocal


def _migrate():
    """SQLite 轻量迁移:给 accounts 表补齐新增列(幂等)"""
    from sqlalchemy import inspect, text
    engine = get_engine()
    try:
        cols = {c["name"] for c in inspect(engine).get_columns("accounts")}
    except Exception:
        return  # 表不存在时跳过
    with engine.begin() as conn:
        if "concurrency_limit" not in cols:
            conn.execute(text(
                "ALTER TABLE accounts ADD COLUMN "
                "concurrency_limit INTEGER NOT NULL DEFAULT 2"))
        if "delete_conversation_after" not in cols:
            conn.execute(text(
                "ALTER TABLE accounts ADD COLUMN "
                "delete_conversation_after BOOLEAN NOT NULL DEFAULT 0"))


def create_all():
    """创建所有表(开发环境用)"""
    from . import models  # noqa: F401  ensure models are registered
    Base.metadata.create_all(bind=get_engine())
    _migrate()


def drop_all():
    Base.metadata.drop_all(bind=get_engine())


@contextmanager
def session_scope() -> Iterator[Session]:
    """上下文管理器,自动 commit/rollback"""
    s = get_session_factory()()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def get_db() -> Iterator[Session]:
    """FastAPI Depends 用"""
    s = get_session_factory()()
    try:
        yield s
    finally:
        s.close()
