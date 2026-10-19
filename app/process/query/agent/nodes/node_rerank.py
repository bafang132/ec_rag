"""
重排节点（node_rerank）：Cross-Encoder/LLM 打分 + 动态 TopK 截断。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query import rerank_documents
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_rerank")
def node_rerank(state: QueryGraphState) -> QueryGraphState:
    """对融合候选做精确相关性重排。"""
    add_running_task(state["session_id"], "node_rerank", state.get("is_stream"))
    state["reranked_docs"] = rerank_documents(state)
    add_done_task(state["session_id"], "node_rerank", state.get("is_stream"))
    return state
