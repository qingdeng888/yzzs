"""
server/api/health.py - 健康检查
"""
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session
from ..db import get_db
from ..models import Account, ApiKey, Usage
from ..auth import require_admin

router = APIRouter()


@router.get("/healthz")
async def healthz():
    return {"status": "ok"}


@router.get("/api/status")
async def status(db: Session = Depends(get_db), _=Depends(require_admin)):
    total_accounts = db.query(func.count(Account.id)).scalar() or 0
    active_accounts = db.query(func.count(Account.id)).filter(Account.status == "active").scalar() or 0
    total_keys = db.query(func.count(ApiKey.id)).scalar() or 0
    active_keys = db.query(func.count(ApiKey.id)).filter(ApiKey.status == "active").scalar() or 0
    return {
        "accounts": {"total": total_accounts, "active": active_accounts},
        "api_keys": {"total": total_keys, "active": active_keys},
    }
