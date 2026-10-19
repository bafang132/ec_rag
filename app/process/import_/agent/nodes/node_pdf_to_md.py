"""
PDF 转 Markdown 节点（node_pdf_to_md）：MinerU 云端优先，本地 pypdf 兜底。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import pdf_to_markdown
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_pdf_to_md")
def node_pdf_to_md(state: ImportGraphState) -> ImportGraphState:
    """PDF → Markdown。"""
    add_running_task(state["task_id"], "node_pdf_to_md")
    state = pdf_to_markdown(state)
    add_done_task(state["task_id"], "node_pdf_to_md")
    return state
