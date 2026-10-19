"""
对象存储门面模块：文档图片存储（MinIO 优先，本地静态文件兜底）。

未配置 MinIO 时，图片保存在 output/<日期>/<task_id>/images/ 下，
由查询服务的 /api/files 路由提供 HTTP 访问。
"""
from pathlib import Path
from typing import Optional

from app.shared.config.minio_config import minio_config
from app.shared.clients import minio_utils
from app.shared.runtime.logger import logger


class ObjectStorageGateway:
    """对象存储门面。"""

    @property
    def enabled(self) -> bool:
        return minio_config.enabled

    def upload_image(self, local_path: str, object_name: str) -> Optional[str]:
        """
        上传图片，返回可访问 URL。

        MinIO 可用时返回 MinIO 公网 URL；否则返回本地静态路由 URL
        （/api/files/<object_name>，由查询服务映射到本地文件）。
        """
        if self.enabled:
            url = minio_utils.upload_image(local_path, object_name)
            if url:
                return url
            logger.info("MinIO 上传失败，回退本地静态文件服务")
        return f"/api/files/{object_name}"

    def delete_prefix(self, prefix: str) -> int:
        """删除指定前缀的图片（重新导入前清理）。"""
        if self.enabled:
            return minio_utils.delete_prefix(prefix)
        return 0

    def resolve_local_path(self, object_name: str) -> Optional[Path]:
        """把 object_name 映射到本地文件路径（本地静态服务用）。"""
        # object_name 约定：<日期>/<task_id>/images/<文件名>
        candidate = Path("output") / object_name
        if candidate.exists():
            return candidate
        candidate = Path("app/output") / object_name
        if candidate.exists():
            return candidate
        return None


object_storage_gateway = ObjectStorageGateway()
