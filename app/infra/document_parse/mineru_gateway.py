"""
MinerU 云端 PDF 解析网关：版面/表格/公式还原度最高的 PDF → Markdown 方案。

流程：申请上传地址 → PUT 上传 → 轮询解析状态 → 下载结果 zip → 解压取 Markdown。
超大 PDF（> MINERU_PAGE_LIMIT 页）先在本地按连续页拆分，逐片解析后合并。
"""
import io
import time
import zipfile
from pathlib import Path
from typing import Optional

import requests

from app.shared.config.mineru_config import mineru_config
from app.shared.runtime.logger import logger


def _headers() -> dict:
    return {"Authorization": f"Bearer {mineru_config.api_token}"}


def _upload_file(pdf_path: str) -> str:
    """申请上传地址并上传文件，返回文件标识。"""
    resp = requests.post(
        f"{mineru_config.base_url}/file-urls",
        headers=_headers(),
        json={"file_name": Path(pdf_path).name, "url": True},
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") not in (0, 200):
        raise RuntimeError(f"MinerU 申请上传地址失败：{data}")
    upload_url = data["data"]["file_url"]
    with open(pdf_path, "rb") as f:
        put = requests.put(upload_url, data=f, timeout=300)
        put.raise_for_status()
    return data["data"]["file_id"] if "file_id" in data.get("data", {}) else upload_url.split("/")[-1].split("?")[0]


def _poll_result(file_id: str) -> Optional[str]:
    """轮询解析状态，返回结果 zip 下载地址。"""
    deadline = time.time() + mineru_config.poll_timeout_seconds
    while time.time() < deadline:
        resp = requests.get(
            f"{mineru_config.base_url}/extract/task/{file_id}",
            headers=_headers(),
            timeout=30,
        )
        resp.raise_for_status()
        data = resp.json().get("data", {})
        state = data.get("state", "")
        if state == "done":
            return data.get("full_zip_url") or data.get("zip_url")
        if state in ("failed", "error"):
            raise RuntimeError(f"MinerU 解析失败：{data.get('err_msg', data)}")
        time.sleep(mineru_config.poll_interval_seconds)
    raise TimeoutError(f"MinerU 解析超时（{mineru_config.poll_timeout_seconds}s）")


def _download_and_extract(zip_url: str, output_dir: str, file_title: str) -> dict:
    """下载结果 zip 并解压，优先取与文件同名的 .md。"""
    resp = requests.get(zip_url, timeout=mineru_config.download_timeout_seconds)
    resp.raise_for_status()
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
        zf.extractall(out_dir)

    md_files = list(out_dir.rglob("*.md"))
    if not md_files:
        raise RuntimeError("MinerU 结果中未找到 Markdown 文件")
    # 优先同名文件，其次 full.md，再首个
    target = next((p for p in md_files if p.stem == file_title), None)
    if target is None:
        target = next((p for p in md_files if p.stem.lower() == "full"), None)
    if target is None:
        target = md_files[0]

    images_dir = target.parent / "images"
    return {
        "md_path": str(target),
        "md_content": target.read_text(encoding="utf-8"),
        "images_dir": str(images_dir) if images_dir.is_dir() else None,
    }


def parse_pdf_with_mineru(pdf_path: str, output_dir: str) -> dict:
    """
    MinerU 解析入口（含超大 PDF 按页拆分）。

    :param pdf_path: PDF 路径
    :param output_dir: 输出目录
    :return: {"md_path", "md_content", "images_dir"}
    """
    file_title = Path(pdf_path).stem
    try:
        from pypdf import PdfReader

        page_count = len(PdfReader(pdf_path).pages)
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"PDF 页数读取失败，按单片处理：{exc}")
        page_count = 0

    if 0 < page_count <= mineru_config.page_limit:
        file_id = _upload_file(pdf_path)
        zip_url = _poll_result(file_id)
        return _download_and_extract(zip_url, output_dir, file_title)

    if page_count > mineru_config.page_limit:
        return _parse_oversized_pdf(pdf_path, output_dir, file_title, page_count)

    # 页数未知：直接尝试单片解析
    file_id = _upload_file(pdf_path)
    zip_url = _poll_result(file_id)
    return _download_and_extract(zip_url, output_dir, file_title)


def _parse_oversized_pdf(pdf_path: str, output_dir: str, file_title: str, page_count: int) -> dict:
    """超大 PDF：按 MINERU_PAGE_LIMIT 页本地拆分，逐片解析后合并正文。"""
    from pypdf import PdfReader, PdfWriter

    logger.info(f"PDF 共 {page_count} 页，超过 MinerU 单文件 {mineru_config.page_limit} 页限制，开始拆分")
    reader = PdfReader(pdf_path)
    out_dir = Path(output_dir)
    split_dir = out_dir / "_split"
    split_dir.mkdir(parents=True, exist_ok=True)

    parts: list[str] = []
    for start in range(0, page_count, mineru_config.page_limit):
        end = min(start + mineru_config.page_limit, page_count)
        writer = PdfWriter()
        for i in range(start, end):
            writer.add_page(reader.pages[i])
        part_path = split_dir / f"{file_title}_part{start // mineru_config.page_limit + 1}.pdf"
        with open(part_path, "wb") as f:
            writer.write(f)
        part_dir = out_dir / f"part{start // mineru_config.page_limit + 1}"
        try:
            file_id = _upload_file(str(part_path))
            zip_url = _poll_result(file_id)
            result = _download_and_extract(zip_url, str(part_dir), part_path.stem)
            parts.append(result["md_content"])
        except Exception as exc:  # noqa: BLE001
            logger.error(f"分片解析失败（页 {start}-{end}）：{exc}")

    merged = "\n\n".join(p for p in parts if p)
    md_path = out_dir / f"{file_title}.md"
    md_path.write_text(merged, encoding="utf-8")
    return {"md_path": str(md_path), "md_content": merged, "images_dir": None}
