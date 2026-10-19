"""
查询流程状态定义文件：描述 LangGraph 问答链全程流转的数据结构。
"""
from typing import List
from typing_extensions import TypedDict
import copy


class QueryGraphState(TypedDict):
    """QueryGraphState 定义问答流程中流转的数据结构。"""
    session_id: str                # 会话唯一标识
    original_query: str            # 用户原始问题

    # 检索中间数据
    embedding_chunks: list         # 语义检索召回切片
    hyde_embedding_chunks: list    # HyDE 检索召回切片
    web_search_docs: list          # 联网搜索文档

    # 排序数据
    rrf_chunks: list               # RRF 融合后切片
    reranked_docs: list            # 重排后最终 Top-K 文档

    # 生成数据
    prompt: str                    # 组装好的 Prompt
    answer: str                    # 最终答案

    # 辅助信息
    item_names: List[str]          # 关联知识主体
    rewritten_query: str           # 改写后的问题
    history: list                  # 历史对话（旧→新）
    is_stream: bool                # 是否流式输出
    image_urls: List[str]          # 答案相关图片
    ec_filters: dict               # 领域筛选 {platform, market, category, doc_type, needs_clarification}
    recommendations: list          # 推荐内容（预留）

    # 合规风险
    risk_level: str                # none/medium/high
    risk_types: List[str]          # 命中的风险类型
    risk_warning: str              # 合规提醒文案

    # 联网搜索
    use_web_search: bool           # 是否启用联网搜索
    web_search_available: bool     # 联网搜索是否可用

    # 用户角色：seller(新手卖家)/operator(成熟运营)/expert(风控专家)
    role: str


query_graph_default_state: QueryGraphState = {
    "session_id": "",
    "original_query": "",
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],
    "prompt": "",
    "answer": "",
    "item_names": [],
    "rewritten_query": "",
    "history": [],
    "is_stream": False,
    "image_urls": [],
    "ec_filters": {},
    "recommendations": [],
    "risk_level": "none",
    "risk_types": [],
    "risk_warning": "",
    "use_web_search": False,
    "web_search_available": False,
    "role": "seller",
}


def create_query_default_state(**overrides) -> QueryGraphState:
    """创建问答流程默认状态，支持覆盖字段。"""
    state = copy.deepcopy(query_graph_default_state)
    state.update(overrides)
    return state


def get_query_default_state() -> QueryGraphState:
    """获取干净的默认状态。"""
    return copy.deepcopy(query_graph_default_state)
