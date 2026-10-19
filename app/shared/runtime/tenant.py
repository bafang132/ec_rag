"""
租户上下文模块：多租户隔离的轻量实现。

通过 contextvars 在「单次请求」生命周期内传递当前租户标识，默认值为 "default"。
未启用鉴权时恒为 default，所有读写路径与单租户版本完全一致，行为与旧版零差异。

隔离边界（按租户命名空间）：
- 本地向量库：data/vector_store/<tenant>/ec_local_db
- 本地历史记录：data/history_<tenant>.json
- 文档版本记录：data/versions.json 内按 "<tenant>/<item_name>" 分桶
"""
from contextvars import ContextVar

DEFAULT_TENANT = "default"

# 请求级租户标识；未设置时回落 default，保证单租户路径不受影响。
_current_tenant: ContextVar[str] = ContextVar("ec_tenant", default=DEFAULT_TENANT)


def get_tenant() -> str:
    """返回当前上下文的租户标识。"""
    return _current_tenant.get()


def set_tenant(value: str) -> None:
    """设置当前上下文的租户标识（为空或非法时回落 default）。"""
    _current_tenant.set(value or DEFAULT_TENANT)


def reset_tenant() -> None:
    """重置为默认租户。"""
    _current_tenant.set(DEFAULT_TENANT)
