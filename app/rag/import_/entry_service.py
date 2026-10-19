"""
导入入口服务模块：文件类型识别与状态初始化（node_entry 的业务实现）。

把任意受支持文件分流为三条路径：
1. Markdown：直接进入图片处理
2. PDF：先转 Markdown（MinerU 优先，本地 pypdf 兜底）
3. 可转换格式（docx/txt/html/xlsx/pptx/...）：markitdown 本地转 Markdown
"""
from pathlib import Path

from app.rag.import_.config import (
    CONVERTIBLE_EXTENSIONS,
    LEGACY_BINARY_EXTENSIONS,
    MARKDOWN_EXTENSIONS,
    PDF_EXTENSIONS,
    PLAIN_TEXT_EXTENSIONS,
)
from app.shared.runtime.logger import logger
from app.shared.utils.path_util import ensure_dir

FILE_TYPE_MARKDOWN = "markdown"
FILE_TYPE_PDF = "pdf"
FILE_TYPE_CONVERTIBLE = "convertible"
FILE_TYPE_PLAIN_TEXT = "plain_text"
FILE_TYPE_LEGACY_BINARY = "legacy_binary"
FILE_TYPE_UNSUPPORTED = "unsupported"


def classify_file(file_path: str) -> str:
    """
    按扩展名识别文件类型。

    :param file_path: 文件路径
    :return: 文件类型标识（见本模块常量）
    """
    suffix = Path(file_path).suffix.lower()
    if suffix in MARKDOWN_EXTENSIONS:
        return FILE_TYPE_MARKDOWN
    if suffix in PDF_EXTENSIONS:
        return FILE_TYPE_PDF
    if suffix in PLAIN_TEXT_EXTENSIONS:
        return FILE_TYPE_PLAIN_TEXT
    if suffix in CONVERTIBLE_EXTENSIONS:
        return FILE_TYPE_CONVERTIBLE
    if suffix in LEGACY_BINARY_EXTENSIONS:
        return FILE_TYPE_LEGACY_BINARY
    return FILE_TYPE_UNSUPPORTED


def resolve_input_file(state: dict) -> dict:
    """
    入口节点业务：校验文件、识别类型、设置路由开关与基础元信息。

    入参 state：local_file_path / task_id / local_dir
    出参 state：file_type / is_md_read_enabled / is_pdf_read_enabled /
                is_convert_enabled / md_path / file_title
    """
    local_file_path = state.get("local_file_path") or ""
    if not local_file_path:
        raise ValueError("local_file_path 不能为空")
    if not Path(local_file_path).exists():
        raise FileNotFoundError(f"导入文件不存在：{local_file_path}")

    file_type = classify_file(local_file_path)
    state["file_type"] = file_type
    state["file_title"] = Path(local_file_path).stem

    # 路由开关（与导入图条件边对应）
    state["is_md_read_enabled"] = file_type == FILE_TYPE_MARKDOWN
    state["is_pdf_read_enabled"] = file_type == FILE_TYPE_PDF
    state["is_convert_enabled"] = file_type in (FILE_TYPE_CONVERTIBLE, FILE_TYPE_PLAIN_TEXT, FILE_TYPE_LEGACY_BINARY)

    # Markdown 文件本身就是最终形态：直接登记 md_path，
    # 后续 node_document_split 的 load_markdown_content 会从该路径读取正文
    # （避免 md_content 为空导致切分节点报错）。
    if file_type == FILE_TYPE_MARKDOWN:
        state["md_path"] = local_file_path
        state["file_title"] = Path(local_file_path).stem

    # 确保任务目录存在
    local_dir = state.get("local_dir")
    if local_dir:
        ensure_dir(Path(local_dir))

    logger.info(
        f"文件类型识别：{Path(local_file_path).name} -> {file_type}"
        f"（md={state['is_md_read_enabled']}, pdf={state['is_pdf_read_enabled']}, convert={state['is_convert_enabled']}）"
    )

    if file_type == FILE_TYPE_UNSUPPORTED:
        raise ValueError(
            f"不支持的文件格式：{Path(local_file_path).suffix}。"
            f"支持：Markdown/PDF/Word/Excel/PPT/TXT/HTML/CSV 等；"
            f"老版 .doc/.ppt/.xls 请先另存为 .docx/.pptx/.xlsx"
        )
    if file_type == FILE_TYPE_LEGACY_BINARY:
        raise ValueError(
            f"老版二进制格式（{Path(local_file_path).suffix}）无法直接解析，"
            f"请在 Word/WPS 中另存为 .docx/.pptx/.xlsx 后重新导入"
        )
    return state
