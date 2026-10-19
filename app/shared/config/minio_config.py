"""
MinIO 配置模块：对象存储（文档图片）。未配置时走本地静态文件。
"""
from dataclasses import dataclass

from app.shared.config.common import env_bool, env_str


@dataclass
class MinioConfig:
    endpoint: str = env_str("MINIO_ENDPOINT", "")
    access_key: str = env_str("MINIO_ACCESS_KEY", "")
    secret_key: str = env_str("MINIO_SECRET_KEY", "")
    bucket_name: str = env_str("MINIO_BUCKET_NAME", "ec-rag-images")
    secure: bool = env_bool("MINIO_SECURE", False)

    @property
    def enabled(self) -> bool:
        """三项核心配置齐全才算启用。"""
        return bool(self.endpoint and self.access_key and self.secret_key)


minio_config = MinioConfig()
