"""
入口节点（node_entry）：校验参数、识别文件类型、设置路由开关。

入参：local_file_path / task_id / local_dir
出参：file_type / is_md_read_enabled / is_pdf_read_enabled / is_convert_enabled / file_title
"""
from app.process.import_.agent.state import ImportGraphState
from app.rag.import_ import resolve_input_file
from app.shared.runtime.logger import logger, node_log
from app.shared.utils.task_utils import add_done_task, add_running_task


@node_log("node_entry")
def node_entry(state: ImportGraphState) -> ImportGraphState:
    """入口节点：识别文件类型并决定流程走向。"""
    add_running_task(state["task_id"], "node_entry")
    state = resolve_input_file(state)
    add_done_task(state["task_id"], "node_entry")
    return state


if __name__ == "__main__":
    # 本地测试：验证 Markdown / PDF / 不支持格式的路由
    from app.process.import_.agent.state import create_default_state

    for path, desc in [
        ("app/source/doc/亚马逊FBA入仓与仓储规则.md", "Markdown"),
        ("data/raw/amazon_rules2.pdf", "PDF"),
        ("data/raw/不支持的格式.xyz", "不支持格式"),
    ]:
        state = create_default_state(task_id=f"test_entry_{desc}", local_file_path=path)
        try:
            result = node_entry(state)
            logger.info(f"[{desc}] file_type={result['file_type']}, "
                        f"md={result['is_md_read_enabled']}, pdf={result['is_pdf_read_enabled']}, "
                        f"convert={result['is_convert_enabled']}")
        except Exception as exc:
            logger.warning(f"[{desc}] 预期内失败：{exc}")
