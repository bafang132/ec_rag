"""
Markdown 图片处理节点（node_md_img）：图片提取 →（可选）视觉摘要 → 上传存储 → 修复引用。
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import enrich_markdown_images
from app.shared.runtime.logger import node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_md_img")
def node_md_img(state: ImportGraphState) -> ImportGraphState:
    """处理 Markdown 中的图片引用。"""
    add_running_task(state["task_id"], "node_md_img")
    state = enrich_markdown_images(state)
    add_done_task(state["task_id"], "node_md_img")
    return state
