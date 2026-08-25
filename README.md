# 云智助手 2api

把云智助手(eaichat.ctyun.cn)的网页端对话转成 OpenAI 兼容 API。

## 功能

- ✅ **OpenAI 兼容接口**:`POST /v1/chat/completions`,支持流式/非流式
- ✅ **多账号轮询**:Web 后台增删改账号,自动 round-robin
- ✅ **Session 缓存**:登录态缓存到 SQLite,避免重复 IAM 登录(降低风控)
- ✅ **API Key 鉴权 + Bearer Token**:支持每日/每月配额
- ✅ **密码保护后台**:bcrypt 哈希存储
- ✅ **错误重试**:网络/5xx/服务器繁忙自动重试,失败账号自动冷却
- ✅ **限流**:每分钟/每天 + 配额
- ✅ **统计**:总请求/成功/失败/Token 全部按 API Key 统计
- ✅ **8 个模型变种**:4 基础模型 + 4 个 `-nothing`(关闭深度思考)变种
- ✅ **深度思考**:默认开启,`<模型>-nothing` 变种关闭;`enable_thinking` 可显式覆盖

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

```bash
cp config.example.yaml config.yaml
# 编辑 config.yaml,改 admin.password 等
```

### 3. 启动

```bash
python run.py
# 或指定配置: python run.py --config /path/to/config.yaml
```

服务启动后:
- API 文档: <http://localhost:8080/docs>
- 后台: <http://localhost:8080/admin/> (默认 `admin` / `changeme`)
- 健康检查: <http://localhost:8080/healthz>

### 4. 添加账号

打开后台 → 账号管理 → 添加账号(账号格式 `ty_xxx#1970740`,密码是你登录用的密码)。
添加后会自动触发首次登录。

### 5. 创建 API Key

后台 → API Keys → 创建 → 拿到 `sk-xxx...`(只显示一次,妥善保存)。

### 6. 客户端调用

#### 用 curl
```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "glm-5.2",
    "messages": [{"role": "user", "content": "你好"}],
    "stream": true
  }'
```

#### 用 OpenAI Python SDK
```python
from openai import OpenAI

client = OpenAI(
    api_key="sk-xxx",
    base_url="http://localhost:8080/v1",
)

resp = client.chat.completions.create(
    model="glm-5.2",
    messages=[{"role": "user", "content": "你好"}],
    stream=True,
)
for chunk in resp:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
```

#### 深度思考
```python
resp = client.chat.completions.create(
    model="qwen-3.7",
    messages=[{"role": "user", "content": "9.11 和 9.8 哪个大?"}],
    stream=True,
    extra_body={"enable_thinking": True},  # 启用深度思考
)
for chunk in resp:
    delta = chunk.choices[0].delta
    rc = getattr(delta, "reasoning_content", None)
    if rc: print(f"[思考] {rc}", end="", flush=True)
    if delta.content: print(delta.content, end="", flush=True)
```

## 支持的模型

| 名称 | keyModel | 说明 |
|------|----------|------|
| `deepseek-v4` | TEXT_DEEPSEEK_V4 | DeepSeek-V4 百万上下文 |
| `glm-5.2` | TEXT_GLM_5.2 | 智谱 GLM-5.2 长任务 |
| `qwen-3.7` | TEXT_QWEN_3.7 | 通义千问 Qwen3.7 |
| `qwen-3-32b` | TEXT_A13 | 通义千问 Qwen3-32B |

> 每个模型默认**开启深度思考**(响应含 `reasoning_content`)。将任意模型名加 `-nothing` 后缀(如 `deepseek-v4-nothing`)即为关闭深度思考的变种。`extra_body.enable_thinking` 仍可显式覆盖(优先级:`-nothing` 变种 > `enable_thinking` 参数 > 默认开启)。

## API

### `POST /v1/chat/completions`

标准 OpenAI 请求体,扩展字段:
- `enable_thinking` (bool): 启用深度思考(默认 true;`-nothing` 变种强制 false)
- `web_search` (bool): 启用联网搜索(默认 true)

标准响应,`reasoning_content` 放在 `delta.reasoning_content`(o1 风格)。

### `GET /v1/models`

返回支持的模型列表。

### 后台路由

| 路由 | 说明 |
|------|------|
| `/admin/login` | 登录 |
| `/admin/` | 仪表盘(总览 + 24h 流量 + Top 5 + 账号状态) |
| `/admin/accounts` | 账号增删改 + 启停 + 刷新 session |
| `/admin/api-keys` | API Key 增删改 + 配额 |
| `/admin/usage` | 详细使用记录(分页 + 按 Key 过滤) |

## 部署

### Docker Compose(推荐)

1. **准备环境文件**
   ```bash
   cp .env.example .env
   # 编辑 .env,务必修改 CTYUN_ADMIN_PASSWORD,建议固定 CTYUN_ADMIN_SECRET
   ```

2. **构建并启动**
   ```bash
   docker compose up -d --build
   ```

3. **访问**
   - 后台: `http://<服务器IP>:10087/admin`
   - API 文档: `http://<服务器IP>:10087/docs`
   - 健康检查: `http://<服务器IP>:10087/healthz`
   - 若宿主机 10087 被占用,改 `.env` 里 `YZZS_PORT`(如 `11088`)再 `docker compose up -d`

**持久化**
- SQLite 数据库与日志保存在命名卷 `yzzs-data`(挂载到容器 `/app/data`)
- `docker compose down` / `docker compose up -d` 重建容器**不会丢数据**
- 仅当需要彻底清除数据时才删卷:`docker volume ls | grep yzzs` 找到实际卷名后 `docker volume rm <卷名>`

**日常运维**
- 查看日志: `docker compose logs -f`
- 升级重建: `docker compose up -d --build`
- 状态/健康: `docker compose ps`(Status 应为 `healthy`)
- 停止: `docker compose down`

### 简单后台运行

```bash
nohup python run.py > /var/log/ctyun-2api.log 2>&1 &
```

### systemd

```ini
# /etc/systemd/system/ctyun-2api.service
[Unit]
Description=ctyun-2api
After=network.target

[Service]
Type=simple
User=ctyun
WorkingDirectory=/opt/ctyun-2api
ExecStart=/opt/ctyun-2api/venv/bin/python run.py
Restart=on-failure
RestartSec=5
Environment=CTYUN_ADMIN_PASSWORD=your_password

[Install]
WantedBy=multi-user.target
```

### Nginx 反向代理 + HTTPS

```nginx
server {
    listen 443 ssl;
    server_name api.example.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://127.0.0.1:8080;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_buffering off;  # 重要:流式响应必须关
        proxy_read_timeout 300s;
    }
}
```

## 数据存储

- 数据库:`./data/ctyun.db` (SQLite)
- 日志:`./data/ctyun.log`
- 账号密码:AES-128 加密(密钥派生自 `admin.session_secret`)
- Session:明文存 SQLite(本地),不加密(部署时建议限制文件权限)

## 故障排除

### 后台显示账号状态 `active` 但聊天返回 "all accounts failed"
- 看 `last_error` 字段
- 在账号管理点 "刷新" 强制重新登录
- 密码可能错误,去账号管理重新添加

### 一直 429 (rate limit)
- 调高 `api.rate_limit.requests_per_minute` 和 `requests_per_day`
- 或在 API Key 里把 daily_quota 调大

### 流式响应卡住
- Nginx 必须 `proxy_buffering off;`
- 看 `data/ctyun.log` 错误

## 风险与限制

1. **不是官方 API**:本项目逆向网页端协议,云智可能随时改版
2. **限流**:云智对单账号有频次限制(实测 IAM 短时间不能重复登录)
3. **多实例**:目前单进程设计,需要时再扩展
4. **Token 估算**:服务端不返回 token,程序粗略估算(中文字符较多时偏大)

## License

仅供学习交流,请勿用于商业用途。
