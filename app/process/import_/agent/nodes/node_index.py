"""
入库节点（node_index）：切片写入向量库（Milvus / 本地库经门面自动选择）。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import index_chunks
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_index")
def node_index(state: ImportGraphState) -> ImportGraphState:
    """切片入库。"""
    add_running_task(state["task_id"], "node_index")
    state = index_chunks(state)
    add_done_task(state["task_id"], "node_index")
    return state
