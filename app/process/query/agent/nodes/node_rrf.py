"""
RRF 融合节点（node_rrf）：多路召回结果融合排序。
"""
from app.process.query.agent.state import QueryGraphState
from app.rag.query import fuse_retrieval_results
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_rrf")
def node_rrf(state: QueryGraphState) -> QueryGraphState:
    """对多路召回做 RRF 融合。"""
    add_running_task(state["session_id"], "node_rrf", state.get("is_stream"))
    state["rrf_chunks"] = fuse_retrieval_results(state)
    add_done_task(state["session_id"], "node_rrf", state.get("is_stream"))
    return state
