"""
run.py - 启动入口
用法: python run.py [--config config.yaml]
"""
import argparse
import logging
import sys
import uvicorn

from server.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./config.yaml", help="配置文件路径")
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None, type=int)
    parser.add_argument("--reload", action="store_true", help="开发模式热重载")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.server.workers != 1:
        raise SystemExit("当前轻量 SQLite/内存状态模式仅支持 workers=1")

    host = args.host or cfg.server.host
    port = args.port or cfg.server.port

    # 配置日志
    log_level = cfg.server.log_level.upper()
    handlers = [logging.StreamHandler(sys.stdout)]
    if cfg.logging.file:
        try:
            handlers.append(logging.FileHandler(cfg.logging.file, encoding="utf-8"))
        except Exception as e:
            print(f"无法写日志文件 {cfg.logging.file}: {e}")
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
        force=True,
    )

    log = logging.getLogger("ctyun-2api")
    log.info(f"启动服务 {host}:{port}, 数据目录: {cfg.data_dir}")
    log.info(f"DB: {cfg.database.url}")

    uvicorn.run(
        "server.app:create_app",
        host=host,
        port=port,
        factory=True,
        log_level=log_level.lower(),
        reload=args.reload,
        workers=1 if args.reload else cfg.server.workers,
    )


if __name__ == "__main__":
    main()
