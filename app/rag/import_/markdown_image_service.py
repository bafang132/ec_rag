"""
Markdown 图片处理服务模块：提取被引用图片 →（可选）视觉摘要 → 上传存储 → 修复引用。

- 未配置 VL 模型/密钥时，跳过摘要，仅做图片提取与引用修复；
- MinIO 未配置时经对象存储门面走本地静态文件（/api/files/<object_name>）。
"""
import re
import shutil
from pathlib import Path
from typing import Optional

from app.infra.object_storage import object_storage_gateway
from app.rag.import_.config import SUPPORTED_IMAGE_EXTENSIONS
from app.shared.runtime.load_prompt import load_prompt
from app.shared.runtime.logger import logger

# Markdown 图片引用：![alt](src)
_IMG_PATTERN = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")
_CONTEXT_CHARS = 100


def scan_images(md_content: str, md_path: str) -> list[dict]:
    """
    扫描 Markdown 中真正被引用的本地图片，并截取前后上下文。

    :return: [{alt, src, abs_path, before, after}]（仅本地存在且扩展名受支持的图片）
    """
    md_dir = Path(md_path).parent
    images: list[dict] = []
    for match in _IMG_PATTERN.finditer(md_content or ""):
        alt, src = match.group(1), match.group(2).strip()
        if src.startswith(("http://", "https://", "data:")):
            continue
        abs_path = (md_dir / src).resolve()
        if not abs_path.exists() or abs_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue
        start = max(0, match.start() - _CONTEXT_CHARS)
        images.append({
            "alt": alt,
            "src": src,
            "abs_path": str(abs_path),
            "before": (md_content[start:match.start()] or "").strip(),
            "after": (md_content[match.end():match.end() + _CONTEXT_CHARS] or "").strip(),
        })
    logger.info(f"扫描到被引用的本地图片 {len(images)} 张")
    return images


def summarize_image(image: dict) -> str:
    """
    用视觉模型为图片生成文字摘要（VL 模型不可用时返回 alt 或空串）。

    摘要会替换 Markdown 中的 alt 文字，让检索/问答能感知图片内容。
    """
    try:
        from app.infra.llm import llm_provider
        from app.shared.model.lm_utils import invoke_text, build_messages

        client = llm_provider.vision_chat()
        prompt = load_prompt(
            "image_summary",
            image_content=(image.get("before", ""), image.get("after", "")),
        )
        summary = invoke_text(client, build_messages("你是文档图片理解助手", prompt)).strip()
        if summary:
            return summary[:200]
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"图片摘要生成失败，使用 alt 兜底：{exc}")
    return (image.get("alt") or "").strip()[:200]


def upload_images_and_replace(state: dict, images: list[dict]) -> dict:
    """
    上传图片到对象存储（或本地静态目录），把 Markdown 中的本地引用替换为可访问 URL，
    并把 alt 文字替换为图片摘要。
    """
    if not images:
        return state

    md_content = state["md_content"]
    task_id = state.get("task_id", "unknown")
    replaced = 0

    for image in images:
        summary = summarize_image(image)
        # 对象名约定：<task_id>/images/<文件名>（与本地静态目录结构一致）
        object_name = f"{task_id}/images/{Path(image['abs_path']).name}"
        url = object_storage_gateway.upload_image(image["abs_path"], object_name)

        # 本地静态服务兜底时，确保文件在 output/<object_name> 可达
        if url and url.startswith("/api/files/"):
            target = Path("output") / object_name
            try:
                target.parent.mkdir(parents=True, exist_ok=True)
                if not target.exists():
                    shutil.copy2(image["abs_path"], target)
            except OSError as exc:
                logger.warning(f"图片复制到静态目录失败：{exc}")

        alt_text = summary or image.get("alt") or Path(image["abs_path"]).stem
        old_ref = f"![{image['alt']}]({image['src']})"
        new_ref = f"![{alt_text}]({url})"
        if old_ref in md_content:
            md_content = md_content.replace(old_ref, new_ref)
            replaced += 1

    state["md_content"] = md_content
    # 备份原文件（_new.md 后缀）
    try:
        md_path = Path(state["md_path"])
        backup = md_path.with_name(md_path.stem + "_new.md")
        backup.write_text(md_content, encoding="utf-8")
        state["md_path"] = str(md_path)
    except OSError as exc:
        logger.warning(f"图片处理后 Markdown 备份失败：{exc}")
    logger.info(f"图片引用替换完成：{replaced}/{len(images)}")
    return state


def enrich_markdown_images(state: dict) -> dict:
    """
    图片处理主入口（node_md_img 业务实现）。

    仅当存在 images 目录（PDF 解析产出）或 Markdown 内联本地图片时才处理。
    """
    md_path = state.get("md_path") or ""
    md_content = state.get("md_content") or ""
    if not md_path or not md_content:
        logger.info("无 Markdown 内容，跳过图片处理")
        return state

    images = scan_images(md_content, md_path)
    if not images:
        logger.info("未发现需要处理的本地图片引用")
        return state
    return upload_images_and_replace(state, images)
