"""
PDF → Markdown 服务模块（node_pdf_to_md 业务实现）：经文档解析门面走
MinerU 云端解析（配置后）或本地 pypdf 提取，并做归一化清洗。
"""
from pathlib import Path

from app.infra.document_parse import document_parse_gateway
from app.rag.import_.document_convert_service import normalize_markdown
from app.shared.runtime.logger import logger


def pdf_to_markdown(state: dict) -> dict:
    """
    PDF → Markdown 主入口。

    :param state: 导入图状态（local_file_path / local_dir / file_title）
    :return: 更新了 md_path / md_content / images_dir 的状态
    """
    pdf_path = state["local_file_path"]
    file_title = state.get("file_title") or Path(pdf_path).stem
    local_dir = state.get("local_dir") or str(Path(pdf_path).parent)

    result = document_parse_gateway.pdf_to_markdown(pdf_path, local_dir)

    md_content = normalize_markdown(result.get("md_content") or "")
    if not md_content:
        raise ValueError(
            f"PDF 解析结果为空：{pdf_path}。"
            f"请确认 PDF 为文字版（扫描件无法提取文本，需 OCR）"
        )

    # 统一把 md 落到任务目录，便于溯源与备份
    md_path = str(Path(local_dir) / f"{file_title}.md")
    if result.get("md_path") and Path(result["md_path"]).exists():
        try:
            Path(md_path).write_text(md_content, encoding="utf-8")
        except OSError as exc:
            logger.warning(f"Markdown 落盘失败，使用解析器输出路径：{exc}")
            md_path = result["md_path"]
    else:
        Path(md_path).write_text(md_content, encoding="utf-8")

    state["md_path"] = md_path
    state["md_content"] = md_content
    state["images_dir"] = result.get("images_dir")
    logger.info(f"PDF 转 Markdown 完成：{file_title}，{len(md_content)} 字符")
    return state
