"""
导入质量测试：文件分类、Markdown 切分（Python/C 路径）、元数据兜底、编码探测、格式归一化。
"""
import json

import pytest

from app.rag.import_.config import (
    CHUNK_MAX_SIZE,
    CHUNK_SIZE,
    TEXT_FILE_ENCODINGS,
)
from app.rag.import_.document_convert_service import (
    normalize_markdown,
    read_plain_text,
)
from app.rag.import_.entry_service import classify_file
from app.rag.import_.metadata_service import (
    build_fallback_metadata,
    normalize_metadata,
)
from app.rag.import_.split_service import refine_chunks, split_by_titles, split_document
from app.shared.utils.c_accel import c_text_splitter, c_split
from app.shared.utils.text_split import split_text


class TestFileClassification:
    """文件类型识别。"""

    @pytest.mark.parametrize("path,expected", [
        ("doc.md", "markdown"),
        ("doc.markdown", "markdown"),
        ("doc.pdf", "pdf"),
        ("doc.docx", "convertible"),
        ("doc.html", "convertible"),
        ("doc.xlsx", "convertible"),
        ("doc.pptx", "convertible"),
        ("note.txt", "plain_text"),
        ("data.csv", "plain_text"),
        ("legacy.doc", "legacy_binary"),
        ("legacy.xls", "legacy_binary"),
        ("unknown.xyz", "unsupported"),
    ])
    def test_classify(self, path, expected):
        assert classify_file(path) == expected


class TestTextSplit:
    """文本切分（纯 Python 路径 + C 加速路径）。"""

    SAMPLE = (
        "# 亚马逊FBA仓储规则\n\n"
        "月度仓储费按立方英尺每月收取，旺季费率上调。"
        "长期仓储费针对库龄超过365天的库存收取。\n\n"
        "IPI分数低于500将面临仓储容量限制，建议定期清理滞销库存。"
    )

    def test_python_split_basic(self):
        chunks = split_text(self.SAMPLE, chunk_size=80, chunk_overlap=10)
        assert len(chunks) >= 2
        assert all(len(c) <= 100 for c in chunks)  # 允许标题前缀
        assert all(c.strip() for c in chunks)

    def test_python_split_overlap(self):
        chunks = split_text("一二三四五六七八九十" * 10, chunk_size=15, chunk_overlap=5)
        assert len(chunks) > 1

    def test_python_split_empty(self):
        assert split_text("") == []
        assert split_text("   \n  ") == []

    def test_c_split_matches_python_when_available(self):
        """C 库加载成功时结果应与 Python 实现语义一致（数量一致、内容覆盖）。"""
        c_text_splitter.ensure_loaded()
        if not c_text_splitter.enabled:
            pytest.skip("C 加速库未编译/未通过自检（预期：默认走 Python）")
        c_chunks = c_split(self.SAMPLE, chunk_size=80, chunk_overlap=10)
        py_chunks = split_text(self.SAMPLE, chunk_size=80, chunk_overlap=10)
        assert len(c_chunks) == len(py_chunks)

    def test_c_lib_self_test_safe(self):
        """C 库自检：不可用时必须安全禁用而非崩溃（关键安全属性）。"""
        # 无论 DLL 是否存在/是否损坏，ensure_loaded 都不应抛异常
        result = c_text_splitter.ensure_loaded()
        assert isinstance(result, bool)
        # 之后 split 始终可用（自动降级）
        assert c_split("测试文本。第二句。", chunk_size=10, chunk_overlap=2)

    def test_split_by_titles(self):
        chunks = split_by_titles(self.SAMPLE, "测试文档")
        assert len(chunks) >= 1
        assert all(c["file_title"] == "测试文档" for c in chunks)
        assert any("仓储规则" in c["title"] for c in chunks)

    def test_split_by_titles_code_block(self):
        """代码块内的 # 不应被当作标题。"""
        md = "# 真标题\n\n```python\n# 这不是标题\nprint(1)\n```\n\n正文内容"
        chunks = split_by_titles(md, "t")
        titles = [c["title"] for c in chunks]
        assert "# 真标题" in titles
        assert not any("这不是标题" in t for t in titles)

    def test_refine_chunks_long_short(self):
        sections = [
            {"title": "长章节", "content": "很长的内容。" * 500, "file_title": "t"},
            {"title": "短章节", "content": "短", "file_title": "t"},
        ]
        refined = refine_chunks(sections, max_len=CHUNK_MAX_SIZE, min_len=CHUNK_SIZE)
        # 长章节被切分，短章节保留（单章节不跨标题合并）
        assert any("长章节-" in c["title"] for c in refined)
        assert all(len(c["content"]) <= CHUNK_MAX_SIZE + 100 for c in refined)

    def test_split_document_end_to_end(self, tmp_path):
        md_path = tmp_path / "测试文档.md"
        md_path.write_text(self.SAMPLE, encoding="utf-8")
        state = {
            "md_path": str(md_path),
            "md_content": self.SAMPLE,
            "file_title": "测试文档",
        }
        result = split_document(state)
        chunks = result["chunks"]
        assert len(chunks) >= 1
        # chunks.json 备份生成
        assert (tmp_path / "chunks.json").exists()
        backup = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
        assert len(backup) == len(chunks)


class TestMetadataFallback:
    """元数据规则兜底。"""

    def test_fallback_by_filename(self):
        meta = build_fallback_metadata("亚马逊FBA入仓与仓储规则详解内容", "亚马逊FBA入仓与仓储规则")
        assert meta["platform"] == "亚马逊"
        assert meta["category"] == "物流仓储"

    def test_fallback_by_content(self):
        meta = build_fallback_metadata("欧洲VAT税率为19%，需注册税号后申报", "税务文档")
        assert meta["market"] == "欧洲"
        assert meta["category"] == "税务与财务"

    def test_normalize_merges_model_and_fallback(self):
        state = {"md_content": "FBA仓储费内容", "local_file_path": "/x/亚马逊FBA入仓与仓储规则.md"}
        meta = normalize_metadata(
            {"platform": "亚马逊", "market": "北美", "category": "物流仓储", "doc_type": "实操教程"},
            state, "亚马逊FBA入仓与仓储规则",
        )
        assert meta["platform"] == "亚马逊"
        assert meta["category"] == "物流仓储"
        assert meta["doc_type"] == "实操教程"
        assert meta["source_file"] == "亚马逊FBA入仓与仓储规则.md"

    def test_normalize_invalid_values_fallback(self):
        state = {"md_content": "亚马逊FBA仓储费按立方英尺收取", "local_file_path": ""}
        meta = normalize_metadata(
            {"platform": "不存在的平台", "category": "乱填的类目"},
            state, "亚马逊FBA入仓与仓储规则",
        )
        # 非法值回退到规则推断
        assert meta["platform"] == "亚马逊"
        assert meta["category"] == "物流仓储"


class TestMarkdownNormalize:
    """Markdown 归一化。"""

    def test_newline_unify(self):
        assert normalize_markdown("a\r\nb\rc") == "a\nb\nc"

    def test_html_comment_removed(self):
        assert "注释" not in normalize_markdown("正文<!-- 这是注释 -->结束")

    def test_long_base64_removed(self):
        long_uri = "data:image/png;base64," + "A" * 500
        text = f"![图]({long_uri})"
        result = normalize_markdown(text)
        assert "AAAA" not in result
        assert "[图片]" in result

    def test_blank_lines_compressed(self):
        result = normalize_markdown("a\n\n\n\n\n\nb")
        assert "\n\n\n" not in result

    def test_empty_input(self):
        assert normalize_markdown("") == ""


class TestPlainTextEncoding:
    """纯文本编码探测。"""

    def test_utf8(self, tmp_path):
        p = tmp_path / "utf8.txt"
        p.write_text("中文内容 UTF-8", encoding="utf-8")
        assert read_plain_text(str(p)) == "中文内容 UTF-8"

    def test_gbk(self, tmp_path):
        p = tmp_path / "gbk.txt"
        p.write_bytes("中文内容 GBK".encode("gbk"))
        content = read_plain_text(str(p))
        assert "中文内容" in content

    def test_bom_utf8(self, tmp_path):
        p = tmp_path / "bom.txt"
        p.write_bytes("中文内容".encode("utf-8-sig"))
        content = read_plain_text(str(p))
        assert "中文内容" in content
        assert not content.startswith("\ufeff")
