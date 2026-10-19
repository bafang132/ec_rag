"""
文档转换节点（node_doc_to_md）：docx/txt/html/xlsx/pptx 等 → Markdown。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import convert_to_markdown
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_doc_to_md")
def node_doc_to_md(state: ImportGraphState) -> ImportGraphState:
    """可转换格式 → Markdown（markitdown / 纯文本直读）。"""
    add_running_task(state["task_id"], "node_doc_to_md")
    state = convert_to_markdown(state)
    add_done_task(state["task_id"], "node_doc_to_md")
    return state
