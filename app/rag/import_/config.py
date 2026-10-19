"""
导入链配置模块：集中管理切块、元数据抽取、向量化与入库相关策略参数。
"""
# ---------------- 切块参数 ----------------
# CHUNK_SIZE 作为「短块合并阈值」，CHUNK_MAX_SIZE 作为「超长章节切分阈值」
CHUNK_SIZE = 600
CHUNK_MAX_SIZE = 1000
CHUNK_OVERLAP = 20

# ---------------- 向量化参数 ----------------
# 智谱 Embedding 单请求上限 64 条，留余量按 60 批（分批调用实践结论）
EMBEDDING_BATCH_SIZE = 60
# 元数据语义增强拼接时最大术语数
VECTOR_ENHANCE_MAX_TERMS = 6

# ---------------- 元数据抽取参数 ----------------
METADATA_CONTEXT_CHUNK_K = 5
METADATA_CONTEXT_TOTAL_MAX_CHARS = 10000

# ---------------- MinerU 参数 ----------------
MINERU_MODEL_VERSION = "vlm"
MINERU_POLL_TIMEOUT_SECONDS = 600
MINERU_POLL_INTERVAL_SECONDS = 3
MINERU_DOWNLOAD_TIMEOUT_SECONDS = 30
MINERU_PAGE_LIMIT = 200

# ---------------- 文档格式分流 ----------------
# 导入链统一归一化原则：任意格式 -> Markdown -> 复用同一条 Markdown 处理链。
MARKDOWN_EXTENSIONS = {".md", ".markdown", ".mdown", ".mkd"}
PDF_EXTENSIONS = {".pdf"}
CONVERTIBLE_EXTENSIONS = {
    ".docx", ".txt", ".text", ".log", ".html", ".htm", ".xhtml", ".mhtml", ".xml",
    ".xlsx", ".xlsm", ".csv", ".tsv", ".pptx", ".json", ".yaml", ".yml",
    ".ipynb", ".epub", ".rst", ".org", ".rtf",
}
# 纯文本类扩展名：直接按编码探测读取，不经过 markitdown（更快且编码可控）
PLAIN_TEXT_EXTENSIONS = {".txt", ".text", ".log", ".csv", ".tsv", ".rst", ".org"}
# 老版二进制格式 markitdown 无法解析，需用户另存为新格式后再导入
LEGACY_BINARY_EXTENSIONS = {".doc", ".ppt", ".xls", ".wps", ".et", ".dps"}

# 纯文本读取时依次尝试的编码，全部失败时以 errors="ignore" 兜底
TEXT_FILE_ENCODINGS = ("utf-8-sig", "utf-8", "gbk", "gb18030", "big5", "latin-1")

# Markdown 归一化：内联 base64 图片超过该长度即丢弃，避免污染切分与向量化
INLINE_DATA_URI_MAX_LENGTH = 200

# ---------------- 图片处理 ----------------
SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"}

# ---------------- 向量库字段长度 ----------------
MILVUS_DEFAULT_VARCHAR_MAX_LENGTH = 512
MILVUS_CHUNK_CONTENT_MAX_LENGTH = 65535
MILVUS_METADATA_VARCHAR_MAX_LENGTH = 512
MILVUS_SCENE_VARCHAR_MAX_LENGTH = 2048
MILVUS_VECTOR_DIM = 1024

# ---------------- 切片入库必填字段兜底 ----------------
CHUNK_FIELD_DEFAULTS: dict = {
    "file_title": "",
    "title": "",
    "parent_title": "",
    "part": 1,
    "content": "",
    "platform": "未分类",
    "market": "未分类",
    "category": "未分类",
    "doc_type": "未分类",
    "article_title": "",
    "author": "未分类",
    "source_file": "",
    "source_path": "",
}
