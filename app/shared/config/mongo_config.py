"""
MongoDB 配置模块：会话与聊天记录持久化。未配置时用本地 JSON 存储。
"""
from dataclasses import dataclass

from app.shared.config.common import env_str


@dataclass
class MongoConfig:
    url: str = env_str("MONGO_URL", "")
    db_name: str = env_str("MONGO_DB_NAME", "ec_rag")

    @property
    def enabled(self) -> bool:
        return bool(self.url)


mongo_config = MongoConfig()
