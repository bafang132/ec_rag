"""
应用基础配置模块：导入服务与查询服务的启动配置。
"""
import os
from dataclasses import dataclass

from dotenv import load_dotenv

from app.shared.config.common import env_bool, env_int, env_list, env_str

load_dotenv()


@dataclass
class AppSettings:
    import_app_name: str = env_str("IMPORT_APP_NAME", "跨境电商知识库导入服务")
    query_app_name: str = env_str("QUERY_APP_NAME", "跨境电商RAG智能问答服务")
    app_env: str = env_str("APP_ENV", "dev")
    app_host: str = env_str("APP_HOST", "0.0.0.0")
    import_app_port: int = env_int("IMPORT_APP_PORT", 8000)
    query_app_port: int = env_int("QUERY_APP_PORT", 8001)
    cors_origins: tuple[str, ...] = tuple(env_list("CORS_ORIGINS", ["*"]))
    # 多租户与权限（默认关闭，关闭时行为与单租户完全一致）
    auth_enabled: bool = env_bool("AUTH_ENABLED", False)
    api_keys: tuple[str, ...] = tuple(env_list("API_KEYS", []))  # 形如 "k1,k2:tenantA"
    default_tenant: str = env_str("DEFAULT_TENANT", "default")


settings = AppSettings()
