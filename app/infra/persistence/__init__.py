"""
持久化门面模块：会话与聊天记录存取。

底层存储自动选择：MongoDB（配置 MONGO_URL）⇄ 本地 JSON 文件。
"""
from app.shared.clients.history_store import get_history_store
from app.shared.runtime.logger import logger


class HistoryRepository:
    """历史记录仓储门面（与各历史存储后端接口一致）。"""

    def save_message(self, session_id: str, role: str, text: str,
                     rewritten_query: str = "", item_names=None, image_urls=None) -> None:
        """保存一条聊天消息。"""
        get_history_store().save_message(
            session_id=session_id, role=role, text=text,
            rewritten_query=rewritten_query, item_names=item_names, image_urls=image_urls,
        )

    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        """读取会话最近消息（时间正序，旧→新）。"""
        return get_history_store().list_recent(session_id, limit=limit)

    def clear_session(self, session_id: str) -> int:
        """清空会话全部消息。"""
        return get_history_store().clear_session(session_id)

    def list_sessions(self, limit: int = 50) -> list[dict]:
        """列出会话摘要（多会话侧边栏）。"""
        return get_history_store().list_sessions(limit=limit)

    def create_session(self, session_id: str) -> bool:
        """创建会话元数据。"""
        return get_history_store().create_session(session_id)

    def remove_session(self, session_id: str) -> int:
        """删除会话及其全部消息。"""
        return get_history_store().remove_session(session_id)

    def rename_session(self, session_id: str, title: str) -> bool:
        """重命名会话。"""
        return get_history_store().rename_session(session_id, title)


history_repository = HistoryRepository()
