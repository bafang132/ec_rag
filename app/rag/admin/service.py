"""
知识库管理服务模块：后台管理所需的主体列表、切片查看与删除能力。
"""
from app.infra.vectorstore import vectorstore_gateway
from app.rag.import_.index_service import remove_old_chunks
from app.shared.runtime.logger import logger


def list_items(limit: int = 100) -> list[dict]:
    """列出知识库所有主体（item_name）及元信息（管理页）。"""
    return vectorstore_gateway.list_items(limit=limit)


def delete_item(item_name: str) -> dict:
    """
    删除指定主体下的所有切片。

    :return: {"success": bool, "deleted_count": int}
    """
    try:
        before = len(vectorstore_gateway.get_chunks(item_name, limit=100000))
        deleted = remove_old_chunks(item_name)
        deleted = deleted or before
        logger.info(f"删除主体[{item_name}]成功，切片数：{deleted}")
        return {"success": True, "deleted_count": deleted}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"删除主体[{item_name}]失败：{exc}", exc_info=True)
        return {"success": False, "error": str(exc)}


def delete_chunk(chunk_id) -> dict:
    """按切片 ID 删除单个切片。"""
    try:
        deleted = vectorstore_gateway.delete_by_chunk_id(chunk_id)
        return {"success": deleted > 0, "deleted_count": deleted}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"删除切片[{chunk_id}]失败：{exc}", exc_info=True)
        return {"success": False, "error": str(exc)}


def get_chunks(item_name: str, limit: int = 20) -> list[dict]:
    """查看指定主体的切片详情。"""
    return vectorstore_gateway.get_chunks(item_name, limit=limit)
