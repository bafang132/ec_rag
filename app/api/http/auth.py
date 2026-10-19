"""
API Key 鉴权中间件（可选启用）。

设计原则：
- 默认关闭（AUTH_ENABLED != true）。关闭时完全透明，行为与旧版一致，零依赖、零侵入。
- 启用后，除公开路径（页面 / 文档 / 健康检查）外，所有接口要求携带合法 API Key：
    Authorization: Bearer <key>   或   X-API-Key: <key>
- 每个 Key 可绑定租户（格式 "key:tenant"，逗号分隔多个；未指定则绑定 default）。
  鉴权成功后把对应租户写入请求级上下文，驱动向量库 / 历史 / 版本按租户隔离。

这样实现「同一套代码，关闭即单机单租户、开启即多租户 + 权限」。
"""
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse

from app.shared.config.common import env_bool, env_list
from app.shared.runtime.tenant import DEFAULT_TENANT, set_tenant

# ---------------- 配置 ----------------
AUTH_ENABLED = env_bool("AUTH_ENABLED", False)

# API_KEYS 形如 "k1, k2:tenantA, k3:tenantB"；未带冒号的 key 绑定 default 租户。
API_KEYS: dict[str, str] = {}
for _part in env_list("API_KEYS", [], sep=","):
    _part = _part.strip()
    if not _part:
        continue
    if ":" in _part:
        _k, _t = _part.split(":", 1)
        API_KEYS[_k.strip()] = _t.strip() or DEFAULT_TENANT
    else:
        API_KEYS[_part] = DEFAULT_TENANT

# ---------------- 公开路径白名单 ----------------
# 页面、Swagger、健康检查永远放行，保证 UI 可打开、探针可用。
_PUBLIC_EXACT = {"/", "/health", "/docs", "/openapi.json", "/redoc"}
_PUBLIC_PREFIX = ("/html", "/docs", "/openapi", "/redoc", "/static", "/api/files")


def _is_public(path: str) -> bool:
    if path in _PUBLIC_EXACT:
        return True
    if path.endswith("/html"):
        return True
    return any(path.startswith(p) for p in _PUBLIC_PREFIX)


def _extract_key(request) -> str | None:
    """从 Authorization: Bearer 或 X-API-Key 头提取 key。"""
    auth = request.headers.get("authorization") or request.headers.get("Authorization")
    if auth:
        scheme, _, token = auth.partition(" ")
        if scheme.lower() == "bearer" and token.strip():
            return token.strip()
        if token.strip():
            return token.strip()
    key = request.headers.get("x-api-key") or request.headers.get("X-API-Key")
    return key.strip() if key and key.strip() else None


class AuthMiddleware(BaseHTTPMiddleware):
    """API Key 鉴权中间件：放行公开路径，其余路径校验 Key 并写入租户上下文。"""

    async def dispatch(self, request, call_next):
        path = request.url.path
        if not AUTH_ENABLED or _is_public(path):
            set_tenant(DEFAULT_TENANT)
            return await call_next(request)

        key = _extract_key(request)
        if key and key in API_KEYS:
            set_tenant(API_KEYS[key])
            return await call_next(request)

        set_tenant(DEFAULT_TENANT)
        return JSONResponse(
            status_code=401,
            content={
                "code": 401,
                "error": "unauthorized",
                "message": "缺少有效的 API Key（使用 Authorization: Bearer <key> 或 X-API-Key 头）",
            },
        )


def setup_auth(app) -> None:
    """为 FastAPI 应用装配鉴权中间件（幂等：重复调用只会挂一份）。"""
    # 仅当启用时才挂，避免任何额外开销与默认行为变化。
    if not AUTH_ENABLED:
        return
    app.add_middleware(AuthMiddleware)
