"""
文档版本管理服务模块：记录每个知识主体的导入版本，支持增量更新。

能力：
- 计算文件指纹（SHA-256），同一主体内容未变化则跳过重索引（增量更新）。
- 记录每次成功导入的版本历史：文件指纹、分片数、来源、时间戳。
- 按租户隔离（key 形如 "<tenant>/<item_name>"）。

持久化：data/versions.json（纯 JSON，非 pickle，安全可审计）。
"""
import hashlib
import json
import threading
import time
from pathlib import Path

from app.shared.runtime.logger import logger
from app.shared.runtime.tenant import DEFAULT_TENANT, get_tenant
from app.shared.utils.path_util import DATA_DIR

_LOCAL_PATH = DATA_DIR / "versions.json"
_lock = threading.Lock()


def _tenant_key(item_name: str) -> str:
    tenant = get_tenant() or DEFAULT_TENANT
    return f"{tenant}/{item_name}"


def _load() -> dict:
    if not _LOCAL_PATH.exists():
        return {}
    try:
        return json.loads(_LOCAL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        logger.warning("版本记录文件损坏，已重置")
        return {}


def _save(data: dict) -> None:
    _LOCAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp = _LOCAL_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(_LOCAL_PATH)


def file_hash(path: str | Path) -> str:
    """计算文件 SHA-256 指纹（流式读取，适合大文件）。"""
    h = hashlib.sha256()
    p = Path(path)
    if not p.exists():
        return ""
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def get_latest_hash(item_name: str) -> str:
    """返回该主体最近一次导入的文件指纹（无记录返回空串）。"""
    with _lock:
        data = _load()
    rec = data.get(_tenant_key(item_name))
    if not rec:
        return ""
    return (rec.get("current") or {}).get("hash", "")


def is_unchanged(item_name: str, content_hash: str) -> bool:
    """内容指纹与上次一致则视为未变化（用于增量跳过）。"""
    return bool(content_hash) and get_latest_hash(item_name) == content_hash


def record_version(item_name: str, content_hash: str, chunk_count: int,
                   source: str = "") -> dict:
    """记录一次成功导入版本，返回版本记录。"""
    version = {
        "hash": content_hash,
        "chunk_count": chunk_count,
        "source": source,
        "ts": time.time(),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with _lock:
        data = _load()
        key = _tenant_key(item_name)
        prev = data.get(key, {})
        history = prev.get("history", []) if isinstance(prev, dict) else []
        history.append(version)
        # 仅保留最近 20 个版本，避免无限膨胀
        history = history[-20:]
        data[key] = {"current": version, "history": history}
        _save(data)
    logger.info(f"记录文档版本：主体[{item_name}] 分片={chunk_count} hash={content_hash[:8]}...")
    return version


def list_versions(item_name: str, limit: int = 20) -> list[dict]:
    """返回该主体的版本历史（新→旧）。"""
    with _lock:
        data = _load()
    rec = data.get(_tenant_key(item_name))
    if not rec:
        return []
    return list(reversed(rec.get("history", [])[-limit:]))
