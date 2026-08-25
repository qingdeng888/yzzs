"""
server/api/models.py - /v1/models
"""
from fastapi import APIRouter, Depends
import time

from ..auth import require_api_key

router = APIRouter()

# 与 chat.py 同步: 基础模型 + -nothing(关闭深度思考)变种
_MODEL_BASE = ["glm-5.2", "deepseek-v4", "qwen-3.7", "qwen-3-32b"]
_CREATED = int(time.time()) - 86400 * 30

MODELS = [
    {"id": mid, "object": "model", "created": _CREATED, "owned_by": "ctyun"}
    for mid in _MODEL_BASE
] + [
    {"id": mid + "-nothing", "object": "model", "created": _CREATED, "owned_by": "ctyun",
     "description": "关闭深度思考的变种"}
    for mid in _MODEL_BASE
]


@router.get("/v1/models")
async def list_models(_: None = Depends(require_api_key)):
    return {"object": "list", "data": MODELS}
