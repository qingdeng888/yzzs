"""
server/crypto.py - 对称加密工具(用于加密存储账号密码)
用 Fernet (AES-128-CBC + HMAC)
"""
from __future__ import annotations
import base64
import hashlib
from typing import Optional
from cryptography.fernet import Fernet, InvalidToken


def _derive_key(secret: str) -> bytes:
    """从任意字符串派生 32 字节 base64 key(给 Fernet)"""
    digest = hashlib.sha256(secret.encode("utf-8")).digest()
    return base64.urlsafe_b64encode(digest)


_fernet: Optional[Fernet] = None


def init_crypto(secret: str):
    """用配置里的 session_secret 派生加密密钥"""
    global _fernet
    _fernet = Fernet(_derive_key(secret))


def encrypt(plaintext: str) -> str:
    if _fernet is None:
        raise RuntimeError("crypto not initialized, call init_crypto() first")
    return _fernet.encrypt(plaintext.encode("utf-8")).decode("ascii")


def decrypt(token: str) -> str:
    if _fernet is None:
        raise RuntimeError("crypto not initialized")
    try:
        return _fernet.decrypt(token.encode("ascii")).decode("utf-8")
    except InvalidToken as e:
        raise ValueError(f"decrypt failed: {e}")
