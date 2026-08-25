"""ToolForge FastAPI application."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any, Dict, Optional

from fastapi import Depends, FastAPI, Header, Request
from fastapi.responses import JSONResponse

from app import __version__
from .auth import require_client_auth
from .config import AppConfig, default_config_path, load_config
from .adapters.anthropic import handle_count_tokens, handle_messages
from .adapters.gemini import handle_generate
from .adapters.openai_chat import handle_chat_completions
from .adapters.openai_responses import handle_responses
from .upstream.router import UpstreamRouter
from .util.metrics import GLOBAL_METRICS


def create_app(config: Optional[AppConfig] = None, *, config_path: Optional[str] = None) -> FastAPI:
    if config is None:
        path = config_path or str(default_config_path())
        config = load_config(path)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.config = config
        app.state.router_upstreams = UpstreamRouter(config)
        yield
        await app.state.router_upstreams.aclose()

    app = FastAPI(
        title="ToolForge",
        version=__version__,
        description="Universal LLM tool-calling middleware",
        lifespan=lifespan,
    )
    app.state.config = config
    app.state.router_upstreams = UpstreamRouter(config)

    @app.get("/healthz")
    @app.get("/health")
    async def healthz() -> Dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "upstreams": [u.name for u in config.upstreams],
        }

    @app.get("/metrics")
    async def metrics() -> Dict[str, Any]:
        if not config.features.enable_metrics:
            return {"enabled": False}
        return GLOBAL_METRICS.snapshot()

    @app.get("/v1/models")
    async def list_models(_: str = Depends(require_client_auth)) -> Dict[str, Any]:
        data = []
        seen = set()
        for upstream in config.upstreams:
            for model in upstream.models:
                if model == "*" or model in seen:
                    continue
                seen.add(model)
                data.append(
                    {
                        "id": model,
                        "object": "model",
                        "owned_by": upstream.name,
                        "permission": [],
                    }
                )
        for alias in config.routing.aliases:
            if alias not in seen:
                data.append(
                    {
                        "id": alias,
                        "object": "model",
                        "owned_by": "routing",
                        "permission": [],
                    }
                )
        return {"object": "list", "data": data}

    @app.post("/v1/chat/completions")
    @app.post("/chat/completions")
    async def chat_completions(
        request: Request,
        _: str = Depends(require_client_auth),
        x_toolforge_fc_mode: Optional[str] = Header(default=None, alias="X-ToolForge-FC-Mode"),
    ):
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be object"}})
        return await handle_chat_completions(
            body,
            config=request.app.state.config,
            router=request.app.state.router_upstreams,
            fc_header=x_toolforge_fc_mode,
        )

    @app.post("/v1/messages")
    @app.post("/messages")
    @app.post("/anthropic/v1/messages")
    async def anthropic_messages(
        request: Request,
        _: str = Depends(require_client_auth),
        x_toolforge_fc_mode: Optional[str] = Header(default=None, alias="X-ToolForge-FC-Mode"),
    ):
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be object"}})
        return await handle_messages(
            body,
            config=request.app.state.config,
            router=request.app.state.router_upstreams,
            fc_header=x_toolforge_fc_mode,
        )

    @app.post("/v1/messages/count_tokens")
    @app.post("/anthropic/v1/messages/count_tokens")
    async def anthropic_count_tokens(
        request: Request,
        _: str = Depends(require_client_auth),
    ):
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be object"}})
        return await handle_count_tokens(body)

    @app.post("/v1/responses")
    @app.post("/responses")
    async def responses(
        request: Request,
        _: str = Depends(require_client_auth),
        x_toolforge_fc_mode: Optional[str] = Header(default=None, alias="X-ToolForge-FC-Mode"),
    ):
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be object"}})
        return await handle_responses(
            body,
            config=request.app.state.config,
            router=request.app.state.router_upstreams,
            fc_header=x_toolforge_fc_mode,
        )

    @app.api_route("/v1beta/models/{model_path:path}", methods=["POST", "GET"])
    @app.api_route("/v1/models/{model_path:path}", methods=["POST"])
    async def gemini_route(
        model_path: str,
        request: Request,
        _: str = Depends(require_client_auth),
        x_toolforge_fc_mode: Optional[str] = Header(default=None, alias="X-ToolForge-FC-Mode"),
    ):
        # model_path like "gemini-2.5-pro:generateContent" or "...:streamGenerateContent"
        if ":" not in model_path:
            return JSONResponse(status_code=404, content={"error": {"message": "expected model:method"}})
        model, method = model_path.rsplit(":", 1)
        method = method.strip()
        stream = method == "streamGenerateContent"
        if method not in {"generateContent", "streamGenerateContent"}:
            return JSONResponse(status_code=404, content={"error": {"message": f"unknown method {method}"}})
        body = await request.json()
        if not isinstance(body, dict):
            return JSONResponse(status_code=400, content={"error": {"message": "body must be object"}})
        return await handle_generate(
            body,
            model=model,
            stream=stream,
            config=request.app.state.config,
            router=request.app.state.router_upstreams,
            fc_header=x_toolforge_fc_mode,
        )

    return app


def _load_default_app() -> FastAPI:
    try:
        return create_app()
    except FileNotFoundError:
        from .config import AppConfig, ClientAuthConfig, UpstreamConfig

        return create_app(
            AppConfig(
                client_authentication=ClientAuthConfig(enabled=False),
                upstreams=[
                    UpstreamConfig(
                        name="placeholder",
                        base_url="http://127.0.0.1:9/v1",
                        models=["*"],
                        is_default=True,
                    )
                ],
            )
        )


app = _load_default_app()
