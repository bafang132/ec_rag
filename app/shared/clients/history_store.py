"""
历史记录存储模块：MongoDB（配置后启用）⇄ 本地 JSON 文件（默认）。

统一接口：
- save_message(session_id, role, text, rewritten_query, item_names, image_urls)
- list_recent(session_id, limit) -> list[dict]（时间正序，旧→新）
- clear_session(session_id) -> int
- list_sessions(limit) -> list[dict]
- create_session(session_id) -> bool
- remove_session(session_id) -> int
- rename_session(session_id, title) -> bool
"""
import json
import threading
import time
from pathlib import Path
from typing import Optional

from app.shared.config.mongo_config import mongo_config
from app.shared.runtime.logger import logger
from app.shared.runtime.tenant import DEFAULT_TENANT, get_tenant
from app.shared.utils.path_util import DATA_DIR

_lock = threading.Lock()


def _local_history_path(tenant: str):
    """本地历史文件路径：默认租户沿用 history.json，其余按租户分文件。"""
    if tenant == DEFAULT_TENANT:
        return DATA_DIR / "history.json"
    return DATA_DIR / f"history_{tenant}.json"

# ---------------------- 本地 JSON 实现 ----------------------

class LocalHistoryStore:
    """本地 JSON 文件历史存储（零依赖，单进程适用）。"""

    def __init__(self, tenant: str = DEFAULT_TENANT) -> None:
        self._path = _local_history_path(tenant)

    def _load_local(self) -> dict:
        if not self._path.exists():
            return {"sessions": {}, "messages": []}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            logger.warning("本地历史文件损坏，已重置")
            return {"sessions": {}, "messages": []}

    def _save_local(self, data: dict) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self._path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self._path)

    def save_message(self, session_id: str, role: str, text: str,
                     rewritten_query: str = "", item_names=None, image_urls=None) -> None:
        with _lock:
            data = self._load_local()
            data["messages"].append({
                "session_id": session_id,
                "role": role,
                "text": text,
                "rewritten_query": rewritten_query,
                "item_names": item_names or [],
                "image_urls": image_urls or [],
                "ts": time.time(),
            })
            self._save_local(data)

    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        with _lock:
            data = self._load_local()
        msgs = [m for m in data["messages"] if m.get("session_id") == session_id]
        msgs.sort(key=lambda m: m.get("ts", 0))
        return msgs[-limit:]

    def clear_session(self, session_id: str) -> int:
        with _lock:
            data = self._load_local()
            before = len(data["messages"])
            data["messages"] = [m for m in data["messages"] if m.get("session_id") != session_id]
            deleted = before - len(data["messages"])
            data["sessions"].pop(session_id, None)
            self._save_local(data)
            return deleted

    def list_sessions(self, limit: int = 50) -> list[dict]:
        with _lock:
            data = self._load_local()
        sessions: dict[str, dict] = {}
        for m in sorted(data["messages"], key=lambda m: m.get("ts", 0)):
            sid = m.get("session_id")
            if not sid:
                continue
            entry = sessions.setdefault(sid, {
                "session_id": sid,
                "title": "",
                "message_count": 0,
                "last_ts": 0,
                "last_message": "",
            })
            entry["message_count"] += 1
            entry["last_ts"] = m.get("ts", 0)
            entry["last_message"] = (m.get("text") or "")[:50]
        for sid, meta in data.get("sessions", {}).items():
            if sid in sessions and meta.get("title"):
                sessions[sid]["title"] = meta["title"]
        result = sorted(sessions.values(), key=lambda s: s["last_ts"], reverse=True)
        return result[:limit]

    def create_session(self, session_id: str) -> bool:
        with _lock:
            data = self._load_local()
            data["sessions"][session_id] = {"session_id": session_id, "title": "", "created_at": time.time()}
            self._save_local(data)
            return True

    def remove_session(self, session_id: str) -> int:
        return self.clear_session(session_id)

    def rename_session(self, session_id: str, title: str) -> bool:
        with _lock:
            data = self._load_local()
            data["sessions"].setdefault(session_id, {"session_id": session_id})
            data["sessions"][session_id]["title"] = title
            self._save_local(data)
            return True


# ---------------------- MongoDB 实现 ----------------------

class MongoHistoryStore:
    """MongoDB 历史存储（集合：session_meta / chat_message）。"""

    def __init__(self, tenant: str = DEFAULT_TENANT) -> None:
        from pymongo import MongoClient

        # 多租户：非默认租户使用独立 database 后缀，实现数据隔离
        db_name = mongo_config.db_name
        if tenant and tenant != DEFAULT_TENANT:
            db_name = f"{mongo_config.db_name}_{tenant}"
        self._client = MongoClient(mongo_config.url, serverSelectionTimeoutMS=3000)
        self._db = self._client[db_name]
        self._sessions = self._db["session_meta"]
        self._messages = self._db["chat_message"]
        # 触发一次连接验证，失败则初始化即抛错（由工厂降级）
        self._client.server_info()
        logger.success(f"MongoDB 历史存储初始化成功：db={db_name}")

    def save_message(self, session_id: str, role: str, text: str,
                     rewritten_query: str = "", item_names=None, image_urls=None) -> None:
        self._messages.insert_one({
            "session_id": session_id,
            "role": role,
            "text": text,
            "rewritten_query": rewritten_query,
            "item_names": item_names or [],
            "image_urls": image_urls or [],
            "ts": time.time(),
        })

    def list_recent(self, session_id: str, limit: int = 10) -> list[dict]:
        cursor = (
            self._messages.find({"session_id": session_id})
            .sort("ts", -1)
            .limit(limit)
        )
        msgs = list(cursor)
        msgs.reverse()  # 时间正序（旧→新）
        for m in msgs:
            m.pop("_id", None)
        return msgs

    def clear_session(self, session_id: str) -> int:
        result = self._messages.delete_many({"session_id": session_id})
        self._sessions.delete_one({"session_id": session_id})
        return result.deleted_count

    def list_sessions(self, limit: int = 50) -> list[dict]:
        sessions = list(self._sessions.find().sort("created_at", -1).limit(limit))
        out = []
        for s in sessions:
            sid = s.get("session_id")
            count = self._messages.count_documents({"session_id": sid})
            last = self._messages.find_one({"session_id": sid}, sort=[("ts", -1)])
            out.append({
                "session_id": sid,
                "title": s.get("title", ""),
                "message_count": count,
                "last_ts": last.get("ts", 0) if last else 0,
                "last_message": (last.get("text", "") if last else "")[:50],
            })
        return out

    def create_session(self, session_id: str) -> bool:
        self._sessions.update_one(
            {"session_id": session_id},
            {"$setOnInsert": {"session_id": session_id, "title": "", "created_at": time.time()}},
            upsert=True,
        )
        return True

    def remove_session(self, session_id: str) -> int:
        return self.clear_session(session_id)

    def rename_session(self, session_id: str, title: str) -> bool:
        self._sessions.update_one(
            {"session_id": session_id},
            {"$set": {"title": title}},
            upsert=True,
        )
        return True


# ---------------------- 工厂 ----------------------

# 按租户缓存存储实例：同一租户复用同一后端，跨租户互不干扰。
_instances: dict[str, object] = {}


def get_history_store():
    """
    获取当前租户的历史存储实例（按租户缓存单例）。

    配置了 MongoDB 则优先，连接失败自动降级本地 JSON；默认租户路径与旧版一致。

    :return: 历史存储实例（统一接口）
    """
    tenant = get_tenant()
    if tenant in _instances:
        return _instances[tenant]

    if mongo_config.enabled:
        try:
            _instances[tenant] = MongoHistoryStore(tenant=tenant)
            return _instances[tenant]
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"MongoDB 连接失败，降级本地 JSON 历史存储：{exc}")
    _instances[tenant] = LocalHistoryStore(tenant=tenant)
    logger.info(f"使用本地 JSON 历史存储（tenant={tenant}）")
    return _instances[tenant]
