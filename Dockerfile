# 云智助手 2api - 生产镜像
# 构建: docker build -t yzzs-2api .
FROM python:3.13-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# 先装依赖(利用 Docker 层缓存)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目源码(config.yaml 作为默认配置,运行时由 CTYUN_* 环境变量覆盖)
COPY . .

# 数据目录(SQLite 数据库 + 日志),由命名卷挂载持久化
RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 10088

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:10088/',timeout=3).status==200 else 1)"

CMD ["python", "run.py"]
