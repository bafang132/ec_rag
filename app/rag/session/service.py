"""
会话管理服务模块：多会话的创建、列表、删除与重命名。
"""
from app.infra.persistence import history_repository
from app.shared.runtime.logger import logger


def create_session(session_id: str) -> bool:
    """创建会话元数据（持久化，保证空会话也出现在列表）。"""
    try:
        return history_repository.create_session(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"创建会话失败：{exc}")
        return False


def list_all_sessions(limit: int = 50) -> list[dict]:
    """列出会话摘要（前端多会话侧边栏）。"""
    try:
        return history_repository.list_sessions(limit=limit)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"列出会话失败：{exc}")
        return []


def remove_session(session_id: str) -> int:
    """删除会话及其全部聊天记录。"""
    try:
        return history_repository.remove_session(session_id)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"删除会话失败：{exc}")
        return 0


def rename_session_title(session_id: str, title: str) -> bool:
    """重命名会话。"""
    try:
        return history_repository.rename_session(session_id, title)
    except Exception as exc:  # noqa: BLE001
        logger.error(f"重命名会话失败：{exc}")
        return False
