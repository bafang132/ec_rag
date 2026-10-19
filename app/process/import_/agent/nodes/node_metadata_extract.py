"""
元数据抽取节点（node_metadata_extract）：抽取平台/市场/类目/文档类型并回写切片。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import extract_and_apply_metadata
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_metadata_extract")
def node_metadata_extract(state: ImportGraphState) -> ImportGraphState:
    """抽取跨境电商元数据。"""
    add_running_task(state["task_id"], "node_metadata_extract")
    state = extract_and_apply_metadata(state)
    add_done_task(state["task_id"], "node_metadata_extract")
    return state
