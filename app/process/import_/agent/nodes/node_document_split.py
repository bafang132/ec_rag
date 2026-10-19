"""
文档切分节点（node_document_split）：标题语义粗切 + 长切短合精修（C 加速）。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import split_document
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_document_split")
def node_document_split(state: ImportGraphState) -> ImportGraphState:
    """Markdown → 语义切片。"""
    add_running_task(state["task_id"], "node_document_split")
    state = split_document(state)
    add_done_task(state["task_id"], "node_document_split")
    return state
