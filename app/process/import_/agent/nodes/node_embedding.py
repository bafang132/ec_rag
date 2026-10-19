"""
向量化节点（node_embedding）：批量生成 dense + sparse 向量（语义增强 + 术语归一化）。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import generate_chunk_embeddings
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_embedding")
def node_embedding(state: ImportGraphState) -> ImportGraphState:
    """切片向量化。"""
    add_running_task(state["task_id"], "node_embedding")
    state = generate_chunk_embeddings(state)
    add_done_task(state["task_id"], "node_embedding")
    return state
