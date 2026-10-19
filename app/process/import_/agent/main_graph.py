"""
导入流程主图（main_graph）：组织导入链节点执行顺序与分支流转。

链路：
  node_entry ──条件路由──> node_pdf_to_md ──> node_md_img ──> node_document_split
                        └─> node_doc_to_md ──> node_md_img ──> node_document_split
                        └─> node_md_img（Markdown 直读）
  node_document_split → node_metadata_extract → node_embedding → node_index → END
"""
from dotenv import load_dotenv
from langgraph.graph import END, StateGraph

from app.process.import_.agent.nodes.node_doc_to_md import node_doc_to_md
from app.process.import_.agent.nodes.node_document_split import node_document_split
from app.process.import_.agent.nodes.node_embedding import node_embedding
from app.process.import_.agent.nodes.node_entry import node_entry
from app.process.import_.agent.nodes.node_index import node_index
from app.process.import_.agent.nodes.node_md_img import node_md_img
from app.process.import_.agent.nodes.node_metadata_extract import node_metadata_extract
from app.process.import_.agent.nodes.node_pdf_to_md import node_pdf_to_md
from app.process.import_.agent.state import ImportGraphState
from app.shared.runtime.logger import logger

load_dotenv()

# 1. 定义状态图
workflow = StateGraph(ImportGraphState)

# 2. 添加节点
workflow.add_node("node_entry", node_entry)
workflow.add_node("node_pdf_to_md", node_pdf_to_md)
workflow.add_node("node_doc_to_md", node_doc_to_md)
workflow.add_node("node_md_img", node_md_img)
workflow.add_node("node_document_split", node_document_split)
workflow.add_node("node_metadata_extract", node_metadata_extract)
workflow.add_node("node_embedding", node_embedding)
workflow.add_node("node_index", node_index)

# 3. 入口节点
workflow.set_entry_point("node_entry")


# 4. 入口后条件路由：按文件类型分流
def after_entry_node(state: ImportGraphState):
    if state.get("is_md_read_enabled"):
        return "node_md_img"
    if state.get("is_pdf_read_enabled"):
        return "node_pdf_to_md"
    if state.get("is_convert_enabled"):
        return "node_doc_to_md"
    return END


workflow.add_conditional_edges(
    "node_entry",
    after_entry_node,
    {
        "node_md_img": "node_md_img",
        "node_pdf_to_md": "node_pdf_to_md",
        "node_doc_to_md": "node_doc_to_md",
        END: END,
    },
)

# 5. 静态边：三条解析路径汇合到图片处理，之后统一走 切分→元数据→向量化→入库
workflow.add_edge("node_pdf_to_md", "node_md_img")
workflow.add_edge("node_doc_to_md", "node_md_img")
workflow.add_edge("node_md_img", "node_document_split")
workflow.add_edge("node_document_split", "node_metadata_extract")
workflow.add_edge("node_metadata_extract", "node_embedding")
workflow.add_edge("node_embedding", "node_index")
workflow.add_edge("node_index", END)

# 6. 编译图对象
kb_import_app = workflow.compile()

if __name__ == "__main__":
    import os

    from app.process.import_.agent.state import create_default_state
    from app.shared.utils.path_util import PROJECT_ROOT

    logger.info("===== 导入链全流程测试（Markdown 样例） =====")
    test_md = PROJECT_ROOT / "app" / "source" / "doc" / "亚马逊FBA入仓与仓储规则.md"
    if not test_md.exists():
        logger.error(f"测试文件不存在：{test_md}")
    else:
        state = create_default_state(
            task_id="test_import_workflow_001",
            local_file_path=str(test_md),
            local_dir=str(PROJECT_ROOT / "output" / "test_import"),
            item_name=test_md.stem,
        )
        final_state = None
        for event in kb_import_app.stream(state):
            for node_name in event:
                logger.info(f"节点执行完成：{node_name}")
                final_state = event[node_name]
        if final_state:
            logger.info(
                f"导入完成：切片 {len(final_state.get('chunks', []))} 个，"
                f"入库 {final_state.get('indexed_count', 0)} 个，"
                f"元数据 {final_state.get('ec_metadata', {})}"
            )
