"""
MinerU 配置模块：云端 PDF 解析（版面/表格/公式还原）。
"""
from dataclasses import dataclass

from app.shared.config.common import env_int, env_str


@dataclass
class MineruConfig:
    api_token: str = env_str("MINERU_API_TOKEN", "")
    base_url: str = env_str("MINERU_BASE_URL", "https://mineru.net/api/v4")
    model_version: str = env_str("MINERU_MODEL_VERSION", "vlm")
    poll_timeout_seconds: int = env_int("MINERU_POLL_TIMEOUT_SECONDS", 600)
    poll_interval_seconds: int = env_int("MINERU_POLL_INTERVAL_SECONDS", 3)
    download_timeout_seconds: int = env_int("MINERU_DOWNLOAD_TIMEOUT_SECONDS", 30)
    page_limit: int = env_int("MINERU_PAGE_LIMIT", 200)

    @property
    def enabled(self) -> bool:
        return bool(self.api_token)


mineru_config = MineruConfig()
