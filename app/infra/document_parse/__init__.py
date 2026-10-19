"""
文档解析门面模块：PDF → Markdown（MinerU 云端优先，本地 pypdf 兜底）。

未配置 MINERU_API_TOKEN 时自动使用本地文本提取，保证导入链路始终可用。
"""
from app.shared.config.mineru_config import mineru_config
from app.shared.runtime.logger import logger


class DocumentParseGateway:
    """文档解析门面。"""

    @property
    def mineru_enabled(self) -> bool:
        return mineru_config.enabled

    def pdf_to_markdown(self, pdf_path: str, output_dir: str) -> dict:
        """
        将 PDF 转换为 Markdown。

        :param pdf_path: PDF 文件路径
        :param output_dir: 输出目录（存放 .md 与 images/）
        :return: {"md_path": ..., "md_content": ..., "images_dir": ... 或 None}
        """
        if self.mineru_enabled:
            try:
                from app.infra.document_parse.mineru_gateway import parse_pdf_with_mineru

                result = parse_pdf_with_mineru(pdf_path, output_dir)
                if result and result.get("md_content"):
                    logger.info(f"PDF 经 MinerU 解析成功：{pdf_path}")
                    return result
                logger.warning("MinerU 解析结果为空，降级本地提取")
            except Exception as exc:  # noqa: BLE001
                logger.warning(f"MinerU 解析失败，降级本地提取：{exc}")

        from app.infra.document_parse.local_pdf import extract_pdf_locally

        logger.info(f"PDF 经本地 pypdf 提取：{pdf_path}")
        return extract_pdf_locally(pdf_path, output_dir)


document_parse_gateway = DocumentParseGateway()
