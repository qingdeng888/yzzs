"""
server/config.py - 配置加载(YAML + 环境变量覆盖)
"""
from __future__ import annotations
import os
import secrets
from pathlib import Path
from typing import Optional
import yaml
from pydantic import BaseModel, Field


class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = 8080
    workers: int = 1
    log_level: str = "INFO"


class AdminConfig(BaseModel):
    username: str = "admin"
    password: str = ""  # 首次启动必须显式设置
    session_secret: str = ""  # 留空则自动生成
    session_expire_hours: int = 24
    secure_cookie: bool = True


class DatabaseConfig(BaseModel):
    url: str = "sqlite:///./data/ctyun.db"
    echo: bool = False


class RateLimitConfig(BaseModel):
    requests_per_minute: int = 60
    requests_per_day: int = 5000


class ApiConfig(BaseModel):
    rate_limit: RateLimitConfig = Field(default_factory=RateLimitConfig)
    default_daily_quota: int = 1000
    default_monthly_quota: int = 20000
    retry_max: int = 2
    retry_cooldown_seconds: int = 60
    session_refresh_skew_seconds: int = 300
    acquire_wait_timeout_seconds: int = 15  # 并发槽位满时的排队等待上限,超时返回 503
    enable_web_search_default: bool = True
    # 是否在账号被风控时立即禁用(可手动启用)
    auto_disable_on_iam_40050: bool = True


class LoggingConfig(BaseModel):
    level: str = "INFO"
    file: str = "./data/ctyun.log"
    rotation: str = "20 MB"  # 简化,目前用固定文件


class AppConfig(BaseModel):
    server: ServerConfig = Field(default_factory=ServerConfig)
    admin: AdminConfig = Field(default_factory=AdminConfig)
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)
    api: ApiConfig = Field(default_factory=ApiConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    data_dir: str = "./data"

    def get_session_secret(self) -> str:
        """获取持久化的 session/加密密钥。"""
        secret_file = Path(self.data_dir) / ".session_secret"
        if not self.admin.session_secret and secret_file.exists():
            self.admin.session_secret = secret_file.read_text(encoding="utf-8").strip()
        if not self.admin.session_secret:
            self.admin.session_secret = secrets.token_urlsafe(32)
            secret_file.write_text(self.admin.session_secret + "\n", encoding="utf-8")
            try:
                secret_file.chmod(0o600)
            except OSError:
                pass
        return self.admin.session_secret


def load_config(path: str | Path = "./config.yaml") -> AppConfig:
    """从 YAML 文件加载,环境变量可覆盖部分字段"""
    p = Path(path)
    if not p.exists():
        p = Path("./config.example.yaml")
    if p.exists():
        with open(p, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    else:
        data = {}

    cfg = AppConfig(**data)

    # 环境变量覆盖
    if v := os.environ.get("CTYUN_ADMIN_PASSWORD"):
        cfg.admin.password = v
    if v := os.environ.get("CTYUN_ADMIN_USERNAME"):
        cfg.admin.username = v
    if v := os.environ.get("CTYUN_ADMIN_SECRET"):
        cfg.admin.session_secret = v
    if v := os.environ.get("CTYUN_ADMIN_SECURE_COOKIE"):
        cfg.admin.secure_cookie = v.lower() in {"1", "true", "yes", "on"}
    if v := os.environ.get("CTYUN_DB_URL"):
        cfg.database.url = v
    if v := os.environ.get("CTYUN_HOST"):
        cfg.server.host = v
    if v := os.environ.get("CTYUN_PORT"):
        cfg.server.port = int(v)
    if v := os.environ.get("CTYUN_LOG_LEVEL"):
        cfg.server.log_level = v
    if v := os.environ.get("CTYUN_DATA_DIR"):
        cfg.data_dir = v
        cfg.logging.file = str(Path(v) / "ctyun.log")
        cfg.database.url = f"sqlite:///{Path(v) / 'ctyun.db'}"

    # 确保 data_dir 存在
    Path(cfg.data_dir).mkdir(parents=True, exist_ok=True)

    return cfg
