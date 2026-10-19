"""
导入流程状态定义文件：描述 LangGraph 导入链全程流转的数据结构。
"""
from typing import List
from typing_extensions import TypedDict
import copy


class ImportGraphState(TypedDict):
    """ImportGraphState 定义导入流程中流转的数据结构。"""
    task_id: str                      # 任务唯一标识
    local_file_path: str              # 上传文件的本地路径
    local_dir: str                    # 任务输出目录（md/图片/chunks.json）
    item_name: str                    # 知识主体名（默认取文件名，用于幂等覆盖）

    # 文件类型路由开关
    file_type: str                    # markdown/pdf/convertible/plain_text/legacy_binary/unsupported
    is_md_read_enabled: bool          # Markdown 直读开关
    is_pdf_read_enabled: bool         # PDF 解析开关
    is_convert_enabled: bool          # 格式转换开关（markitdown）

    # 中间产物
    md_path: str                      # Markdown 文件路径
    md_content: str                   # Markdown 正文
    images_dir: str | None            # 图片目录（PDF 解析产出）

    # 切分与向量化
    chunks: List[dict]                # 切片列表（含元数据与向量）
    ec_metadata: dict                 # 文档级元数据（平台/市场/类目/文档类型）
    file_hash: str                    # 源文件指纹（SHA-256，用于增量更新）

    # 结果
    indexed_count: int                # 实际入库切片数
    version_skipped: bool             # 是否因内容未变化而跳过重索引


import_graph_default_state: ImportGraphState = {
    "task_id": "",
    "local_file_path": "",
    "local_dir": "",
    "item_name": "",
    "file_type": "",
    "is_md_read_enabled": False,
    "is_pdf_read_enabled": False,
    "is_convert_enabled": False,
    "md_path": "",
    "md_content": "",
    "images_dir": None,
    "chunks": [],
    "ec_metadata": {},
    "file_hash": "",
    "indexed_count": 0,
    "version_skipped": False,
}


def create_default_state(**overrides) -> ImportGraphState:
    """创建导入流程默认状态，支持覆盖字段。"""
    state = copy.deepcopy(import_graph_default_state)
    state.update(overrides)
    return state


def get_default_state() -> ImportGraphState:
    """获取干净的默认状态。"""
    return copy.deepcopy(import_graph_default_state)
