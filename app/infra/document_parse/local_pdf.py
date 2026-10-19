"""
本地 PDF 文本提取（pypdf）：MinerU 不可用时的兜底方案。

提取每页文本，保留页码信息（拼入 Markdown 标题注释），供答案引用溯源。
"""
from pathlib import Path
from typing import Optional


def extract_pdf_locally(pdf_path: str, output_dir: str) -> dict:
    """
    本地提取 PDF 文本为 Markdown。

    :param pdf_path: PDF 路径
    :param output_dir: 输出目录
    :return: {"md_path", "md_content", "images_dir": None}
    """
    from pypdf import PdfReader

    pdf = Path(pdf_path)
    reader = PdfReader(str(pdf))
    lines: list[str] = [f"# {pdf.stem}", ""]
    for page_no, page in enumerate(reader.pages, start=1):
        text = (page.extract_text() or "").strip()
        if not text:
            continue
        lines.append(f"<!-- page:{page_no} -->")
        lines.append(text)
        lines.append("")
    md_content = "\n".join(lines).strip()

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    md_path = out_dir / f"{pdf.stem}.md"
    md_path.write_text(md_content, encoding="utf-8")
    return {"md_path": str(md_path), "md_content": md_content, "images_dir": None}
