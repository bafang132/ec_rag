"""
文档转换服务模块：非 PDF 文档 → Markdown（markitdown 本地转换 + 纯文本编码探测）。

格式归一化设计：任何来源文档都先统一转换为 Markdown，
后续「图片处理 → 切分 → 元数据抽取 → 向量化 → 入库」只面向 Markdown 一种格式。
"""
import re
from pathlib import Path

from app.rag.import_.config import (
    INLINE_DATA_URI_MAX_LENGTH,
    TEXT_FILE_ENCODINGS,
)
from app.shared.runtime.logger import logger


def normalize_markdown(md_content: str) -> str:
    """
    Markdown 归一化清洗，保证切分/向量化输入干净。

    1. 统一换行符；2. 移除 HTML 注释；3. 剔除超长内联 base64 图片；
    4. 压缩连续空行；5. 清理行尾空白；6. 整体 strip。
    """
    if not md_content:
        return ""
    text = md_content.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"<!--.*?-->", "", text, flags=re.DOTALL)
    # 剔除超长内联 base64 图片（保留短占位符）
    text = re.sub(
        r"!\[[^\]]*\]\(\s*data:image/[^)]+\)",
        lambda m: "[图片]" if len(m.group(0)) > INLINE_DATA_URI_MAX_LENGTH else m.group(0),
        text,
    )
    text = re.sub(r"\n{4,}", "\n\n", text)
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return text.strip()


def read_plain_text(file_path: str) -> str:
    """
    按候选编码依次尝试解码纯文本文件（utf-8-sig → utf-8 → gbk → gb18030 → big5 → latin-1）。

    :param file_path: 文本文件路径
    :return: 解码后的文本
    """
    raw = Path(file_path).read_bytes()
    for encoding in TEXT_FILE_ENCODINGS:
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    logger.warning(f"所有候选编码均失败，以 errors=ignore 兜底：{file_path}")
    return raw.decode("utf-8", errors="ignore")


def convert_via_markitdown(file_path: str) -> str:
    """
    用 markitdown 把 docx/html/xlsx/pptx/epub/rtf 等转换为 Markdown。

    :param file_path: 源文件路径
    :return: Markdown 文本
    """
    try:
        from markitdown import MarkItDown
    except ImportError as exc:
        raise RuntimeError(
            "markitdown 未安装，无法转换该格式。请执行：pip install 'markitdown[all]'"
        ) from exc

    converter = MarkItDown(enable_plugins=False)
    result = converter.convert(file_path)
    return result.text_content or ""


def convert_to_markdown(state: dict) -> dict:
    """
    可转换格式 → Markdown 主入口（node_doc_to_md 对 convertible 分支的业务实现）。

    plain_text 走编码探测直读；其余走 markitdown。结果统一过 normalize_markdown 清洗。
    """
    local_file_path = state["local_file_path"]
    file_title = state.get("file_title") or Path(local_file_path).stem
    local_dir = state.get("local_dir") or str(Path(local_file_path).parent)

    if state.get("file_type") == "plain_text":
        md_content = read_plain_text(local_file_path)
        logger.info(f"纯文本直读完成：{len(md_content)} 字符")
    else:
        md_content = convert_via_markitdown(local_file_path)
        logger.info(f"markitdown 转换完成：{len(md_content)} 字符")

    md_content = normalize_markdown(md_content)
    if not md_content:
        raise ValueError(f"文档转换为空：{local_file_path}")

    md_path = str(Path(local_dir) / f"{file_title}.md")
    Path(md_path).write_text(md_content, encoding="utf-8")

    state["md_path"] = md_path
    state["md_content"] = md_content
    state["images_dir"] = None
    return state
