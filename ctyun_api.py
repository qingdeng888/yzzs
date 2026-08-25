"""
ctyun_api.py - 云智助手 2api 客户端模块

完整的逆向实现:
- 登录: eaiSysInfo + IAM + encryptedTokenAuthorize 三步
- 签名: Web-Signature = SHA256(sort(params) + "&" + MD5(body) + "&" + sk + "&" + ts + "&" + random)
- 对话: OpenAI 兼容的 chat completions

依赖: pip install requests pycryptodome
"""
import requests
import json
import base64
import random
import string
import threading
import urllib.parse
import hashlib
import time
import logging
from typing import Optional, Generator
from datetime import datetime, timedelta, timezone
from Crypto.Cipher import AES
from Crypto.PublicKey import RSA
from Crypto.Cipher import PKCS1_v1_5

log = logging.getLogger("ctyun_api")


# ============== 加密工具 ==============

def b6o_mask(token: str) -> str:
    """B6O masking: standard base64 + character substitution"""
    t = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/="
    a = "1234567890abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ/+="
    b64 = base64.b64encode(token.encode()).decode()
    return ''.join(a[t.index(c)] if c in t else c for c in b64)


def rsa_encrypt(plaintext: str, pubkey_b64: str) -> str:
    """RSAES-PKCS1-V1_5 加密, 返回 hex"""
    pem = f"-----BEGIN PUBLIC KEY-----\n{pubkey_b64}\n-----END PUBLIC KEY-----"
    key = RSA.import_key(pem)
    cipher = PKCS1_v1_5.new(key)
    return cipher.encrypt(plaintext.encode()).hex()


def aes_ecb_decrypt_pkcs7(b64: str, key: str) -> str:
    """AES-ECB-PKCS7 解密, 密钥为任意字符串"""
    data = base64.b64decode(b64)
    cipher = AES.new(key.encode('utf-8'), AES.MODE_ECB)
    pt = cipher.decrypt(data)
    pad = pt[-1]
    if 1 <= pad <= 16:
        pt = pt[:-pad]
    return pt.decode('utf-8')


def decrypt_eai_data(encrypted_b64: str) -> dict:
    """AES-ECB-PKCS7 解密 eaiSysInfo.data, 密钥 = 'chinatelecom@cnn'"""
    key = b"chinatelecom@cnn"
    data = base64.b64decode(encrypted_b64.replace('\r', '').replace('\n', ''))
    cipher = AES.new(key, AES.MODE_ECB)
    pt = cipher.decrypt(data)
    pad = pt[-1]
    if 1 <= pad <= 16:
        pt = pt[:-pad]
    return json.loads(pt)


def gen_client_key(length=16) -> str:
    """生成 16位 可见ASCII (32-126) 随机串"""
    return ''.join(chr(random.randint(32, 126)) for _ in range(length))


def gen_random(length=8) -> str:
    """生成 8位 字母+数字 随机串"""
    chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
    return ''.join(random.choice(chars) for _ in range(length))


def web_sign(params: Optional[dict], data, sk: str) -> dict:
    """
    Web-Signature 计算:
    payload = sortParams(params) + "&" + MD5(data) + "&" + sk + "&" + ts + "&" + random
    signature = SHA256(payload)
    """
    if data is None:
        md5 = ""
    elif isinstance(data, str):
        md5 = hashlib.md5(data.encode()).hexdigest()
    else:
        md5 = hashlib.md5(json.dumps(data, separators=(',', ':')).encode()).hexdigest()

    items = [(k, v) for k, v in (params or {}).items() if v is not None]
    items.sort(key=lambda x: x[0])
    params_str = "&".join(f"{k}={v}" for k, v in items)

    if md5:
        base = f"{params_str}&{md5}" if params_str else md5
    else:
        base = params_str

    ts = str(int(time.time() * 1000))
    rand = gen_random(8)
    payload = f"{base}&{sk}&{ts}&{rand}" if base else f"{sk}&{ts}&{rand}"
    sign = hashlib.sha256(payload.encode()).hexdigest()

    return {"sign": sign, "random": rand, "timestamp": ts}


# ============== 客户端 ==============

class CtyunClient:
    """云智助手 API 客户端"""

    BASE = "https://eaichat.ctyun.cn"
    IAM_BASE = "https://desk.ctyun.cn"
    GATEWAY_BASE = "https://gwyilian.ctyun.cn"

    DEFAULT_CLIENT_KEY = "2153a0480535ec0878874f9091ece1f4d0ac125b4d7f48f83f0095022eb768f514e2"

    def __init__(self, account: str, password: str, user_agent: Optional[str] = None):
        """
        account: 形如 "ty_xxx#1970740" (含 #租户id)
        password: 登录密码
        """
        self.account = account
        self.password = password
        self.user_agent = user_agent or (
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
            '(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36 Edg/151.0.0.0'
        )
        self.session = requests.Session()
        self.session.headers['User-Agent'] = self.user_agent

        # 登录后填充
        self.yl_auth: Optional[str] = None
        self.sk: Optional[str] = None
        self.tenant_id: Optional[int] = None
        self.user_id: Optional[int] = None
        self.xuid = f"pubweb_{''.join(random.choices(string.hexdigits.lower(), k=8))}-{''.join(random.choices(string.hexdigits.lower(), k=4))}-{''.join(random.choices(string.hexdigits.lower(), k=4))}-{''.join(random.choices(string.hexdigits.lower(), k=4))}-{''.join(random.choices(string.hexdigits.lower(), k=12))}"
        self.expires_at: Optional[datetime] = None
        # 客户端级登录锁:串行同一 client 上并发的 401 重登(与 pool 层 entry 锁互补)
        self._login_lock = threading.RLock()

    @classmethod
    def from_session(cls, account: str, password: str, yl_auth: str, sk: str,
                    tenant_id: int, user_id: int, xuid: str = "",
                    expires_at: Optional[datetime] = None) -> "CtyunClient":
        """从已缓存的 session 构造,跳过登录"""
        c = cls(account, password)
        c.yl_auth = yl_auth
        c.sk = sk
        c.tenant_id = tenant_id
        c.user_id = user_id
        if xuid:
            c.xuid = xuid
        c.expires_at = expires_at
        c._logged_in = True
        return c

    @property
    def logged_in(self) -> bool:
        return bool(self.yl_auth and self.sk and self.tenant_id)

    # ---------- 登录流程 ----------

    def _step1_get_ssopk(self) -> tuple:
        """Step 1: 拿 eaiSysInfo, 解密获取 ssopk (RSA公钥)"""
        r = self.session.get(
            f'{self.GATEWAY_BASE}/server/eaiSysInfo',
            headers={'x-eai-tenant-id': '15'}
        )
        r.raise_for_status()
        gateway = decrypt_eai_data(r.json()['data'])
        return gateway['sso']['ssopk'], gateway['sso']['ssopkid']

    def _step2_iam_login(self) -> str:
        """Step 2: IAM 登录, 拿 iamToken (JWT)"""
        device_code = "iam:web" + ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        r = self.session.post(
            f'{self.IAM_BASE}/cloudB/dy/iam/api/auth/iam/login',
            json={
                "userAccount": self.account,
                "password": self.password,
                "deviceCode": device_code,
                "deviceName": "iam:web",
            },
            headers={
                "Content-Type": "application/json",
                "Origin": self.BASE,
            }
        )
        r.raise_for_status()
        j = r.json()
        if j.get('code') != 0:
            raise Exception(f"IAM login failed: {j}")
        return j['data']['token'], j['data']['tenantId'], j['data']['userId']

    def _step3_encrypted_token_authorize(self, iam_token: str, ssopk: str, ssopkid: str) -> tuple:
        """
        Step 3: encryptedTokenAuthorize
        - 生成 clientKey (16位可见ASCII)
        - B6O mask iamToken (作为 eai-iam-token header)
        - RSA 加密 clientKey (作为 clientKey body字段)
        - POST form-urlencoded
        返回: (yl_auth, encrypted_session_key)
        """
        client_key = gen_client_key(16)
        masked_token = b6o_mask(iam_token)
        encrypted_ck = rsa_encrypt(client_key, ssopk)

        body = urllib.parse.urlencode({
            'loginType': 'iamToken',
            'clientId': 'eaiapp',
            'iamEnvCode': '',
            'clientKey': encrypted_ck,
            'clientKeyId': ssopkid,
        })
        r = self.session.post(
            f'{self.BASE}/sso/login/v2/iam/encryptedTokenAuthorize',
            data=body,
            headers={
                'Content-Type': 'application/x-www-form-urlencoded;charset=UTF-8',
                'Origin': self.BASE,
                'Referer': f'{self.BASE}/chat/',
                'eai-iam-token': masked_token,
                'masking-method': 'B6O',
            }
        )
        j = r.json()
        if not j.get('success') or j.get('resultCode') != 0:
            raise Exception(f"encryptedTokenAuthorize failed: {j}")
        return r.headers.get('yl-authorization'), j['data']['sessionKey'], client_key

    def login(self):
        """完整登录流程"""
        ssopk, ssopkid = self._step1_get_ssopk()
        iam_token, tenant_id, user_id = self._step2_iam_login()
        yl_auth, enc_sk, client_key = self._step3_encrypted_token_authorize(
            iam_token, ssopk, ssopkid
        )
        # AES 解密 sessionKey (用 clientKey 作为 key)
        sk = aes_ecb_decrypt_pkcs7(enc_sk, client_key)
        self.yl_auth = yl_auth
        self.sk = sk
        self.tenant_id = tenant_id
        self.user_id = user_id
        # 从 JWT 中解析过期时间
        self.expires_at = self._parse_jwt_exp(yl_auth)
        log.info(f"登录成功 userId={user_id} tenant={tenant_id} expires_at={self.expires_at}")
        return self

    @staticmethod
    def _parse_jwt_exp(jwt_str: str) -> Optional[datetime]:
        """从 yl-authorization JWT 中解析 exp"""
        try:
            parts = jwt_str.split('.')
            if len(parts) < 2:
                return None
            payload_b64 = parts[1] + '=' * (4 - len(parts[1]) % 4)
            payload = json.loads(base64.urlsafe_b64decode(payload_b64).decode('utf-8'))
            if 'exp' in payload:
                return datetime.fromtimestamp(payload['exp'], tz=timezone.utc)
        except Exception as e:
            log.warning(f"parse jwt exp failed: {e}")
        return None

    def is_expired(self, skew_seconds: int = 300) -> bool:
        """检查 session 是否过期(默认提前 5 分钟)"""
        if not self.expires_at:
            return True
        return datetime.now(timezone.utc) >= self.expires_at - timedelta(seconds=skew_seconds)

    def refresh(self):
        """强制刷新 session"""
        self.yl_auth = None
        self.sk = None
        self.expires_at = None
        return self.login()

    # ---------- 业务接口 ----------

    def _request(self, method: str, path: str, params: Optional[dict] = None,
                 data=None, stream: bool = False, _retry_on_expired: bool = True,
                 extra_headers: Optional[dict] = None) -> requests.Response:
        """带 Web-Signature 的请求(自动在 401/403 时重登一次)"""
        if extra_headers is None:
            extra_headers = {}
        if not self.yl_auth:
            raise Exception("Not logged in. Call login() first()")

        url = f'{self.BASE}{path}'
        # 准备 body 字符串(用于签名)
        if data is None:
            body_str = None
        elif isinstance(data, str):
            body_str = data
        else:
            body_str = json.dumps(data, separators=(',', ':'))

        sig = web_sign(params, body_str, self.sk)
        headers = {
            'yl-authorization': self.yl_auth,
            'x-eai-tenant-id': str(self.tenant_id),
            'x-eai-source': 'web-yunphone',
            'x-eai-xuid': self.xuid,
            'x-eai-version': '202060305',
            'x-eai-env': 'pubWeb',
            'x-client-trace-id': ''.join(random.choices(string.hexdigits.lower(), k=32)),
            'yl-product-id': '5',
            'yl-main-version': '202060305',
            'web-signature': sig['sign'],
            'web-random': sig['random'],
            'web-timestamp': sig['timestamp'],
        }
        headers.update(extra_headers)
        if body_str is not None:
            headers['Content-Type'] = 'application/json'
            if stream:
                headers['Accept'] = 'text/event-stream'

        resp = self.session.request(method, url, params=params, data=body_str,
                                     headers=headers, stream=stream, timeout=60)
        # 自动重登: 401/403 / 业务码 401
        if _retry_on_expired and not stream:
            try:
                if resp.status_code in (401, 403):
                    old_auth = self.yl_auth
                    with self._login_lock:
                        # 另一线程可能已重登:auth 变了则跳过重登,直接用新 token 重试
                        if self.yl_auth == old_auth:
                            log.warning(f"auth failed (status={resp.status_code}), refresh and retry")
                            self.refresh()
                    return self._request(method, path, params, data, stream,
                                         _retry_on_expired=False, extra_headers=extra_headers)
            except Exception:
                pass
        return resp

    def query_models(self) -> list:
        """获取模型列表"""
        r = self._request('GET', '/ai/portal/wenc/v2/openai/chat/queryModels',
                          params={'type': 'all'})
        j = r.json()
        return j.get('data', [])

    def remove_conversation(self, conversation_id: str) -> dict:
        """删除指定会话。返回 {'resultCode':0,'resultMsg':'success','data':None}"""
        r = self._request(
            'GET',
            '/ai/portal/wenc/v2/openai/chat/removeConversation',
            params={'conversationId': conversation_id},
            extra_headers={'x-eai-source': 'web-eai'},
        )
        return r.json()

    def clear_all_conversations(self) -> dict:
        """清空该账号所有会话"""
        r = self._request(
            'GET',
            '/ai/portal/wenc/v2/openai/chat/removeConversation',
            params={'isClearAll': 'true'},
            extra_headers={'x-eai-source': 'web-eai'},
        )
        return r.json()

    def chat(self, key_model: str, messages: list, enable_thinking: bool = False,
             web_search: bool = True, stream: bool = True) -> requests.Response:
        """
        对话 (云智原生格式, 不是 OpenAI 标准)
        key_model: e.g. 'TEXT_GLM_5.2'
        messages: [{"role": "user", "content": "...", "verify_id": "uuid", "ref": {"type": "file", "file": []}}]
        """
        # 给 user 消息添加 verify_id 和 ref (如果缺)
        for m in messages:
            if m.get('role') == 'user' and 'verify_id' not in m:
                m['verify_id'] = ''.join(random.choices(string.hexdigits.lower(), k=32))
                m['ref'] = {"type": "file", "file": []}

        body = {
            "key_model": key_model,
            "messages": messages,
            "stream": stream,
            "client_retry": True,
            "web_search": web_search,
            "tenantId": self.tenant_id,
            "enable_thinking": enable_thinking,
        }
        return self._request('POST', '/ai/portal/wenc/v3/openai/chat/completions',
                             data=body, stream=stream)

    # ---------- OpenAI 兼容封装 ----------

    def openai_chat(self, model: str, messages: list, stream: bool = True,
                   enable_thinking: bool = False, **kwargs) -> dict | Generator:
        """
        OpenAI 兼容接口
        model: 用户友好名(如 'deepseek-v4', 'glm-5.2')
        """
        # 模型名映射
        model_map = {
            'deepseek-v4': 'TEXT_DEEPSEEK_V4',
            'glm-5.2': 'TEXT_GLM_5.2',
            'qwen-3.7': 'TEXT_QWEN_3.7',
            'qwen-3-32b': 'TEXT_A13',
        }
        key_model = model_map.get(model.lower(), model)
        if not stream:
            # 非流式: 收集所有 chunk
            r = self.chat(key_model, messages, enable_thinking=enable_thinking, stream=True)
            content = ''
            reasoning = ''
            for line in r.iter_lines():
                if not line: continue
                line = line.decode('utf-8', errors='replace')
                if not line.startswith('data:'): continue
                try:
                    chunk = json.loads(line[5:])
                except: continue
                for choice in chunk.get('choices', []):
                    delta = choice.get('delta', {})
                    content += delta.get('content', '')
                    reasoning += delta.get('reasoning_content', '')
            return {
                'id': 'ctyun-' + str(int(time.time())),
                'object': 'chat.completion',
                'created': int(time.time()),
                'model': model,
                'choices': [{
                    'index': 0,
                    'message': {
                        'role': 'assistant',
                        'content': content,
                        'reasoning_content': reasoning if enable_thinking else None,
                    },
                    'finish_reason': 'stop',
                }],
            }
        else:
            return self._openai_stream(key_model, model, messages, enable_thinking)

    def _openai_stream(self, key_model: str, model: str, messages: list,
                      enable_thinking: bool, on_conversation_id=None) -> Generator:
        """流式生成 OpenAI 格式"""
        r = self.chat(key_model, messages, enable_thinking=enable_thinking, stream=True)
        for line in r.iter_lines():
            if not line: continue
            line = line.decode('utf-8', errors='replace')
            if not line.startswith('data:'): continue
            try:
                chunk = json.loads(line[5:])
            except: continue
            # 首 chunk 携带 conversation_id(该 chunk 无 choices,会被下面 continue 跳过)
            if on_conversation_id is not None:
                cid = chunk.get('conversation_id')
                if cid:
                    on_conversation_id(cid)
                    on_conversation_id = None  # 只取第一个
            delta = chunk.get('choices', [{}])[0].get('delta', {})
            if not delta: continue
            # mojibake 修复
            content = delta.get('content', '')
            reasoning = delta.get('reasoning_content', '')
            if content:
                try:
                    content = content.encode('latin-1').decode('utf-8')
                except:
                    pass
            if reasoning:
                try:
                    reasoning = reasoning.encode('latin-1').decode('utf-8')
                except:
                    pass
            yield {
                'id': chunk.get('id', ''),
                'object': 'chat.completion.chunk',
                'created': chunk.get('created', int(time.time())),
                'model': model,
                'choices': [{
                    'index': 0,
                    'delta': {
                        'role': delta.get('role', 'assistant'),
                        'content': content,
                        'reasoning_content': reasoning if enable_thinking else None,
                    },
                    'finish_reason': chunk.get('choices', [{}])[0].get('finish_reason'),
                }],
            }


# ============== 测试 ==============

if __name__ == '__main__':
    import os
    account = os.environ.get('CTYUN_ACCOUNT')
    password = os.environ.get('CTYUN_PASSWORD')

    if not account or not password:
        print("Set CTYUN_ACCOUNT and CTYUN_PASSWORD env vars")
        exit(1)

    client = CtyunClient(account, password)
    client.login()
    print(f"登录成功: userId={client.user_id} tenant={client.tenant_id}")

    # 模型列表
    models = client.query_models()
    print(f"\n模型列表 ({len(models)} 个):")
    for m in models:
        print(f"  - {m.get('modelName')}: {m.get('keyModel')}")

    # 对话
    print(f"\n对话测试:")
    resp = client.openai_chat(
        model='glm-5.2',
        messages=[{"role": "user", "content": "你好,你是谁?"}],
        stream=True,
    )
    for chunk in resp:
        delta = chunk['choices'][0].get('delta', {})
        c = delta.get('content', '')
        if c:
            print(c, end='', flush=True)
    print()
