"""
查询服务 API 数据模型（适配跨境电商领域设计）。
"""
from typing import List, Optional

from pydantic import BaseModel


class QueryRequest(BaseModel):
    """问答请求体。"""
    query: str                                   # 用户问题
    is_stream: bool = False                      # 是否流式
    session_id: Optional[str] = None             # 会话 ID（空则新建）
    use_web_search: bool = False                 # 是否启用联网搜索
    role: str = "seller"                         # 角色：seller/operator/expert


class QACompatRequest(BaseModel):
    """简化问答请求体（兼容跨境电商 RAG V1 契约：字段名为 question）。"""
    question: str
    role: str = "seller"


class QueryResponse(BaseModel):
    """问答响应体。"""
    message: str = ""
    session_id: str = ""
    answer: str = ""
    done_list: List[str] = []                    # 已完成节点（中文）
    image_urls: List[str] = []


class HistoryItem(BaseModel):
    """历史消息项。"""
    _id: str = ""
    session_id: str = ""
    role: str = ""
    text: str = ""
    rewritten_query: str = ""
    item_names: List[str] = []
    image_urls: List[str] = []
    ts: Optional[float] = None


class HistoryResponse(BaseModel):
    """会话历史响应。"""
    session_id: str = ""
    items: List[HistoryItem] = []


class SessionRenameRequest(BaseModel):
    """会话重命名请求。"""
    title: str


class AdminItemInfo(BaseModel):
    """知识库主体信息（管理页）。"""
    item_name: str = ""
    file_title: str = ""
    title: str = ""
    platform: str = ""
    market: str = ""
    category: str = ""
    doc_type: str = ""
    chunk_count: int = 0
    char_count: int = 0


class AdminListItemsResponse(BaseModel):
    """主体列表响应。"""
    data: List[AdminItemInfo] = []


class AdminChunkItem(BaseModel):
    """切片详情（管理页）。"""
    chunk_id: object = None
    content: str = ""
    title: str = ""
    file_title: str = ""
    item_name: str = ""
    platform: str = ""
    market: str = ""
    category: str = ""
    doc_type: str = ""


class AdminGetChunksResponse(BaseModel):
    """切片查询响应。"""
    item_name: str = ""
    chunks: List[AdminChunkItem] = []
    total: int = 0
