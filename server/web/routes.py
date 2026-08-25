"""
server/web/routes.py - 后台管理路由
"""
from __future__ import annotations
import logging
import secrets
import hmac
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, Depends, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy import func, desc, case
from sqlalchemy.orm import Session

from .. import crypto
from ..auth import (
    ADMIN_COOKIE, get_current_admin, make_admin_token,
    make_api_key, hash_api_key, verify_password,
)
from ..config import AppConfig
from ..db import get_db, session_scope
from ..models import Account, ApiKey, Usage
from .. import session_store
from ..account_pool import get_pool

log = logging.getLogger("web")

router = APIRouter(prefix="/admin", include_in_schema=False)

TEMPLATES_DIR = __file__.replace("routes.py", "templates")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
_login_lock = threading.Lock()
_login_attempts: dict[str, deque[float]] = defaultdict(deque)

def _csrf(request: Request) -> str:
    token = request.cookies.get("ctyun_csrf")
    return token or secrets.token_urlsafe(32)

async def check_csrf(request: Request) -> bool:
    form = await request.form()
    cookie = request.cookies.get("ctyun_csrf")
    token = form.get("csrf_token") or request.headers.get("x-csrf-token")
    if not cookie or not token or not hmac.compare_digest(str(cookie), str(token)):
        return False
    return True

def _login_allowed(client_ip: str) -> bool:
    now = time.time()
    with _login_lock:
        attempts = _login_attempts[client_ip]
        while attempts and now - attempts[0] > 300:
            attempts.popleft()
        return len(attempts) < 10

def _record_login_failure(client_ip: str) -> None:
    with _login_lock:
        _login_attempts[client_ip].append(time.time())


def _cfg(request: Request) -> AppConfig:
    return request.app.state.config


def _render(request: Request, db: Session, name: str, ctx: dict, **kwargs):
    """统一渲染:注入 current_user 供 base.html 导航栏判断"""
    ctx["current_user"] = get_current_admin(request, db)
    ctx["csrf_token"] = _csrf(request)
    response = templates.TemplateResponse(request, name, ctx, **kwargs)
    if not request.cookies.get("ctyun_csrf"):
        response.set_cookie(
            "ctyun_csrf", ctx["csrf_token"], httponly=False,
            secure=_cfg(request).admin.secure_cookie, samesite="lax", path="/",
        )
    return response


def _check_admin(
    request: Request,
    db: Session = Depends(get_db),
) -> Optional[object]:
    """统一鉴权:未登录跳 /admin/login"""
    u = get_current_admin(request, db)
    if u is None:
        return None
    return u


# ============== 认证 ==============

@router.get("/login", response_class=HTMLResponse)
async def login_page(request: Request, msg: Optional[str] = None,
                     db: Session = Depends(get_db)):
    if get_current_admin(request, db) is not None:
        return RedirectResponse("/admin/", status_code=302)
    return _render(request, db, "login.html", {"msg": msg})


@router.post("/login")
async def login_submit(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    cfg = _cfg(request)
    client_ip = request.client.host if request.client else "unknown"
    if not _login_allowed(client_ip):
        raise HTTPException(status_code=429, detail="Too many login attempts")
    from ..models import AdminUser
    u = db.query(AdminUser).filter(AdminUser.username == username).first()
    if not u or not verify_password(password, u.password_hash):
        _record_login_failure(client_ip)
        return _render(
            request, db, "login.html", {"msg": "用户名或密码错误"}, status_code=401
        )
    u.last_login_at = datetime.utcnow()
    db.commit()
    token = make_admin_token(
        username,
        cfg.get_session_secret(),
        cfg.admin.session_expire_hours * 3600,
    )
    resp = RedirectResponse("/admin/", status_code=302)
    resp.set_cookie(
        ADMIN_COOKIE, token,
        max_age=cfg.admin.session_expire_hours * 3600,
        httponly=True, secure=cfg.admin.secure_cookie, samesite="lax", path="/",
    )
    csrf = secrets.token_urlsafe(32)
    resp.set_cookie("ctyun_csrf", csrf, max_age=cfg.admin.session_expire_hours * 3600,
                    httponly=False, secure=cfg.admin.secure_cookie, samesite="lax", path="/")
    return resp


@router.post("/logout")
async def logout(request: Request):
    resp = RedirectResponse("/admin/login", status_code=302)
    resp.delete_cookie(ADMIN_COOKIE, path="/")
    resp.delete_cookie("ctyun_csrf", path="/")
    return resp


# ============== 仪表盘 ==============

@router.get("/", response_class=HTMLResponse)
@router.get("", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        return RedirectResponse("/admin/login", status_code=302)

    total = db.query(func.count(Usage.id)).scalar() or 0
    success = db.query(func.count(Usage.id)).filter(Usage.status == "ok").scalar() or 0
    failed = total - success
    total_prompt = db.query(func.sum(Usage.prompt_tokens)).scalar() or 0
    total_completion = db.query(func.sum(Usage.completion_tokens)).scalar() or 0
    total_reasoning = db.query(func.sum(Usage.reasoning_tokens)).scalar() or 0
    total_latency_sum = db.query(func.sum(Usage.latency_ms)).scalar() or 0
    avg_latency = int(total_latency_sum / success) if success else 0

    since = datetime.utcnow() - timedelta(hours=24)
    rows = db.query(
        func.strftime("%Y-%m-%d %H:00", Usage.created_at).label("hour"),
        func.count(Usage.id).label("n"),
    ).filter(Usage.created_at >= since).group_by("hour").all()
    by_hour = [{"hour": r.hour, "n": r.n} for r in rows]

    top_models = db.query(
        Usage.model, func.count(Usage.id).label("n")
    ).filter(Usage.model != "").group_by(Usage.model).order_by(desc("n")).limit(5).all()

    top_keys = db.query(
        ApiKey.name, ApiKey.id, func.count(Usage.id).label("n")
    ).join(Usage, Usage.api_key_id == ApiKey.id).group_by(ApiKey.id).order_by(desc("n")).limit(5).all()

    accounts = db.query(Account).order_by(Account.id).all()
    api_keys = db.query(ApiKey).order_by(ApiKey.id).all()
    recent = db.query(Usage).order_by(desc(Usage.created_at)).limit(20).all()

    return _render(request, db, "dashboard.html", {
        "total": total, "success": success, "failed": failed,
        "total_prompt": total_prompt,
        "total_completion": total_completion,
        "total_reasoning": total_reasoning,
        "avg_latency": avg_latency,
        "by_hour": by_hour,
        "top_models": top_models,
        "top_keys": top_keys,
        "accounts": accounts,
        "api_keys": api_keys,
        "recent": recent,
    })


# ============== 账号管理 ==============

@router.get("/accounts", response_class=HTMLResponse)
async def accounts_page(request: Request, msg: Optional[str] = None, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        return RedirectResponse("/admin/login", status_code=302)
    accounts = db.query(Account).order_by(Account.id).all()
    return _render(request, db, "accounts.html", {
        "accounts": accounts, "msg": msg,
    })


@router.post("/accounts/add")
async def accounts_add(
    request: Request,
    account: str = Form(...),
    password: str = Form(...),
    remark: str = Form(""),
    concurrency_limit: int = Form(2),
    delete_conversation_after: str = Form(""),
    db: Session = Depends(get_db),
):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    pw_enc = crypto.encrypt(password)
    if db.query(Account).filter(Account.account == account).first():
        return RedirectResponse("/admin/accounts?msg=账号已存在", status_code=302)
    delete_flag = delete_conversation_after in ("1", "on", "true")
    a = session_store.create_account(
        db, account, pw_enc, remark,
        concurrency_limit=concurrency_limit,
        delete_conversation_after=delete_flag,
    )
    aid = a.id
    db.commit()
    pool = get_pool()
    pool.reload()
    pool.set_password(aid, password)
    import threading
    threading.Thread(target=lambda: pool.do_login(aid), daemon=True).start()
    return RedirectResponse("/admin/accounts?msg=已添加", status_code=302)


@router.post("/accounts/{aid}/toggle")
async def accounts_toggle(request: Request, aid: int, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    a = db.get(Account, aid)
    if not a:
        return RedirectResponse("/admin/accounts?msg=账号不存在", status_code=302)
    new_status = "disabled" if a.status == "active" else "active"
    session_store.update_account_status(db, aid, new_status)
    db.commit()
    get_pool().reload()
    return RedirectResponse("/admin/accounts", status_code=302)


@router.post("/accounts/{aid}/refresh")
async def accounts_refresh(request: Request, aid: int, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    pool = get_pool()
    import threading
    def do():
        try:
            pool.do_login(aid)
        except Exception as e:
            log.warning(f"refresh {aid} failed: {e}")
    threading.Thread(target=do, daemon=True).start()
    return RedirectResponse("/admin/accounts?msg=已触发刷新", status_code=302)


@router.post("/accounts/{aid}/delete")
async def accounts_delete(request: Request, aid: int, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    session_store.delete_account(db, aid)
    db.commit()
    get_pool().reload()
    return RedirectResponse("/admin/accounts", status_code=302)


@router.post("/accounts/{aid}/settings")
async def accounts_settings(
    request: Request, aid: int,
    concurrency_limit: int = Form(...),
    delete_conversation_after: str = Form(""),
    db: Session = Depends(get_db),
):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    delete_flag = delete_conversation_after in ("1", "on", "true")
    if not session_store.update_account_settings(db, aid, concurrency_limit, delete_flag):
        return RedirectResponse("/admin/accounts?msg=账号不存在", status_code=302)
    db.commit()
    get_pool().update_account_settings(aid, concurrency_limit, delete_flag)  # 原地更新内存,不触发 reload
    log.info(f"账号 {aid} 更新设置: concurrency_limit={concurrency_limit} delete_conversation_after={delete_flag}")
    return RedirectResponse("/admin/accounts?msg=已更新", status_code=302)


@router.post("/accounts/{aid}/clear-conversations")
async def accounts_clear_conversations(request: Request, aid: int,
                                       db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    pool = get_pool()
    import threading

    def do():
        try:
            msg = pool.clear_conversations(aid)
            log.info(f"账号 {aid} 清空会话结果: {msg}")
        except Exception as e:
            log.warning(f"账号 {aid} 清空会话异常: {e}")

    threading.Thread(target=do, daemon=True).start()
    return RedirectResponse("/admin/accounts?msg=已触发清空会话", status_code=302)


# ============== API Key 管理 ==============

@router.get("/api-keys", response_class=HTMLResponse)
async def apikeys_page(request: Request, msg: Optional[str] = None,
                       new_key: Optional[str] = None, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        return RedirectResponse("/admin/login", status_code=302)
    keys = db.query(ApiKey).order_by(ApiKey.id).all()
    cfg = _cfg(request)
    return _render(request, db, "apikeys.html", {
        "keys": keys, "msg": msg, "new_key": new_key,
        "default_daily_quota": cfg.api.default_daily_quota,
        "default_monthly_quota": cfg.api.default_monthly_quota,
    })


@router.post("/api-keys/add")
async def apikeys_add(
    request: Request,
    name: str = Form(...),
    daily_quota: int = Form(0),
    monthly_quota: int = Form(0),
    remark: str = Form(""),
    db: Session = Depends(get_db),
):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    plain = make_api_key()
    key_hash = hash_api_key(plain)
    prefix = plain[:7]
    suffix = plain[-4:]
    k = ApiKey(
        name=name, key_hash=key_hash, key_prefix=prefix, key_suffix=suffix,
        daily_quota=daily_quota, monthly_quota=monthly_quota, remark=remark,
    )
    db.add(k)
    db.commit()
    return RedirectResponse(f"/admin/api-keys?msg=已创建&new_key={plain}", status_code=302)


@router.post("/api-keys/{kid}/toggle")
async def apikeys_toggle(request: Request, kid: int, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    k = db.get(ApiKey, kid)
    if not k:
        return RedirectResponse("/admin/api-keys?msg=不存在", status_code=302)
    k.status = "disabled" if k.status == "active" else "active"
    db.commit()
    return RedirectResponse("/admin/api-keys", status_code=302)


@router.post("/api-keys/{kid}/delete")
async def apikeys_delete(request: Request, kid: int, db: Session = Depends(get_db)):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    k = db.get(ApiKey, kid)
    if k:
        db.delete(k)
        db.commit()
    return RedirectResponse("/admin/api-keys", status_code=302)


@router.post("/api-keys/{kid}/update")
async def apikeys_update(
    request: Request, kid: int,
    daily_quota: int = Form(...),
    monthly_quota: int = Form(...),
    remark: str = Form(""),
    db: Session = Depends(get_db),
):
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    k = db.get(ApiKey, kid)
    if k:
        k.daily_quota = daily_quota
        k.monthly_quota = monthly_quota
        k.remark = remark
        db.commit()
    return RedirectResponse("/admin/api-keys?msg=已更新", status_code=302)


# ============== 日志统计 ==============

@router.get("/usage", response_class=HTMLResponse)
async def usage_page(request: Request, db: Session = Depends(get_db)):
    """日志统计:按 API Key / 云智账号两个维度聚合 总请求/成功/失败/输入token/输出token"""
    if get_current_admin(request, db) is None:
        return RedirectResponse("/admin/login", status_code=302)

    # 按 API Key 聚合
    by_key = db.query(
        ApiKey.id.label("key_id"),
        ApiKey.name.label("key_name"),
        ApiKey.key_prefix.label("key_prefix"),
        ApiKey.key_suffix.label("key_suffix"),
        func.count(Usage.id).label("total"),
        func.sum(case((Usage.status == "ok", 1), else_=0)).label("success"),
        func.sum(case((Usage.status == "fail", 1), else_=0)).label("failed"),
        func.sum(Usage.prompt_tokens).label("in_tokens"),
        func.sum(Usage.completion_tokens).label("out_tokens"),
    ).outerjoin(Usage, Usage.api_key_id == ApiKey.id).group_by(ApiKey.id).order_by(ApiKey.id).all()

    # 按云智账号聚合
    by_account = db.query(
        Account.id.label("account_id"),
        Account.account.label("account_name"),
        func.count(Usage.id).label("total"),
        func.sum(case((Usage.status == "ok", 1), else_=0)).label("success"),
        func.sum(case((Usage.status == "fail", 1), else_=0)).label("failed"),
        func.sum(Usage.prompt_tokens).label("in_tokens"),
        func.sum(Usage.completion_tokens).label("out_tokens"),
    ).outerjoin(Usage, Usage.account_id == Account.id).group_by(Account.id).order_by(Account.id).all()

    # 最近 50 条明细(折叠展示)
    recent = db.query(Usage).order_by(desc(Usage.created_at)).limit(50).all()
    keys_map = {k.id: k for k in db.query(ApiKey).all()}
    accounts_map = {a.id: a for a in db.query(Account).all()}

    return _render(request, db, "usage.html", {
        "by_key": by_key,
        "by_account": by_account,
        "recent": recent,
        "keys_map": keys_map,
        "accounts_map": accounts_map,
        "msg": request.query_params.get("msg"),
    })


@router.post("/usage/reset-key/{kid}")
async def usage_reset_key(request: Request, kid: int, db: Session = Depends(get_db)):
    """重置某个 API Key 的全部统计记录"""
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    deleted = db.query(Usage).filter(Usage.api_key_id == kid).delete()
    k = db.get(ApiKey, kid)
    name = k.name if k else kid
    db.commit()
    log.info(f"重置 API Key {name} 的统计,删除 {deleted} 条")
    return RedirectResponse(f"/admin/usage?msg=已重置 Key「{name}」的统计({deleted} 条)", status_code=302)


@router.post("/usage/reset-account/{aid}")
async def usage_reset_account(request: Request, aid: int, db: Session = Depends(get_db)):
    """重置某个云智账号的全部统计记录"""
    if get_current_admin(request, db) is None:
        raise HTTPException(401)
    deleted = db.query(Usage).filter(Usage.account_id == aid).delete()
    acc = db.get(Account, aid)
    name = acc.account if acc else aid
    db.commit()
    log.info(f"重置账号 {name} 的统计,删除 {deleted} 条")
    return RedirectResponse(f"/admin/usage?msg=已重置账号「{name}」的统计({deleted} 条)", status_code=302)
