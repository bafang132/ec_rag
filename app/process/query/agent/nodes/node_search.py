"""
混合检索节点（node_search）：语义检索（dense + BM25/sparse）+ 可选联网搜索并发。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query import parallel_search
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_search")
def node_search(state: QueryGraphState) -> QueryGraphState:
    """执行混合向量检索与（可选）联网搜索。"""
    add_running_task(state["session_id"], "node_search", state.get("is_stream"))
    ec_filters = state.get("ec_filters", {}) or {}
    use_web_search = bool(state.get("use_web_search"))

    embedding_chunks, web_docs = parallel_search(
        rewritten_query=state.get("rewritten_query") or state.get("original_query", ""),
        ec_filters=ec_filters,
        scene=ec_filters.get("scene", ""),
        limit=15,
        use_web_search=use_web_search,
    )
    state["embedding_chunks"] = embedding_chunks
    state["web_search_docs"] = web_docs or []
    add_done_task(state["session_id"], "node_search", state.get("is_stream"))
    return state
