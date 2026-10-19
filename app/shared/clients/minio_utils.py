"""
MinIO 客户端工具模块：文档图片上传与访问（未配置时由调用方走本地静态文件）。
"""
from typing import Optional

from app.shared.config.minio_config import minio_config
from app.shared.runtime.logger import logger

_client = None
_bucket_ready = False


def get_minio_client():
    """获取 MinIO 客户端单例。"""
    global _client
    if _client is not None:
        return _client
    if not minio_config.enabled:
        raise ValueError("MinIO 未配置（MINIO_ENDPOINT/ACCESS_KEY/SECRET_KEY）")
    from minio import Minio

    _client = Minio(
        endpoint=minio_config.endpoint,
        access_key=minio_config.access_key,
        secret_key=minio_config.secret_key,
        secure=minio_config.secure,
    )
    logger.success(f"MinIO 连接成功：{minio_config.endpoint}")
    return _client


def ensure_bucket() -> bool:
    """确保 bucket 存在（进程内只检查一次）。"""
    global _bucket_ready
    if _bucket_ready:
        return True
    client = get_minio_client()
    if not client.bucket_exists(minio_config.bucket_name):
        client.make_bucket(minio_config.bucket_name)
        logger.info(f"MinIO bucket 已创建：{minio_config.bucket_name}")
    _bucket_ready = True
    return True


def upload_image(local_path: str, object_name: str) -> Optional[str]:
    """
    上传图片到 MinIO 并返回可访问 URL。

    :param local_path: 本地图片路径
    :param object_name: 对象名（含目录前缀）
    :return: 公开访问 URL；失败返回 None
    """
    try:
        client = get_minio_client()
        ensure_bucket()
        client.fput_object(minio_config.bucket_name, object_name, local_path)
        scheme = "https" if minio_config.secure else "http"
        url = f"{scheme}://{minio_config.endpoint}/{minio_config.bucket_name}/{object_name}"
        logger.debug(f"图片上传 MinIO 成功：{url}")
        return url
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"图片上传 MinIO 失败，将使用本地静态文件：{exc}")
        return None


def delete_prefix(prefix: str) -> int:
    """删除指定前缀下的全部对象（重新导入前清理旧图片）。"""
    try:
        client = get_minio_client()
        ensure_bucket()
        objects = client.list_objects(minio_config.bucket_name, prefix=prefix, recursive=True)
        count = 0
        for obj in objects:
            client.remove_object(minio_config.bucket_name, obj.object_name)
            count += 1
        return count
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"删除 MinIO 前缀失败：{exc}")
        return 0
