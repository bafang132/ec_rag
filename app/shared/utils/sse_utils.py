"""
SSE 工具模块：会话级事件队列与流式生成器。
"""
import asyncio
import json
import queue
import time
from typing import Any, Dict, Optional

from app.shared.runtime.logger import logger


class SSEEvent:
    READY = "ready"                # 连接建立
    PROGRESS = "progress"          # 任务节点进度
    DELTA = "delta"                # LLM 流式输出增量
    FINAL = "final"                # 最终完整答案
    ERROR = "error"                # 错误信息
    CLOSE = "__close__"            # 关闭连接信号
    WEB_SEARCH_HINT = "web_search_hint"  # 本地知识不足，提示用户是否启用联网搜索


# 全局 SSE 会话队列存储：Key=session_id, Value=queue.Queue
_session_stream: Dict[str, "queue.Queue"] = {}
# 队列创建时间，用于惰性回收长期遗留队列（客户端断开但未发 __close__），避免内存泄漏
_session_stream_ts: Dict[str, float] = {}
SSE_QUEUE_TTL_SECONDS = 1800


def get_sse_queue(session_id: str) -> Optional["queue.Queue"]:
    """获取指定 session 的队列。"""
    return _session_stream.get(session_id)


def _prune_stale_queues() -> None:
    """回收超过 TTL 的遗留队列，防止内存泄漏。"""
    now = time.time()
    expired = [sid for sid, ts in _session_stream_ts.items() if now - ts > SSE_QUEUE_TTL_SECONDS]
    for sid in expired:
        _session_stream.pop(sid, None)
        _session_stream_ts.pop(sid, None)
        logger.debug(f"[SSE] 回收过期队列: {sid}")


def create_sse_queue(session_id: str) -> "queue.Queue":
    """创建并注册一个新的 SSE 队列。"""
    logger.debug(f"[SSE] 创建队列 session={session_id}")
    _prune_stale_queues()
    q: "queue.Queue" = queue.Queue()
    _session_stream[session_id] = q
    _session_stream_ts[session_id] = time.time()
    return q


def remove_sse_queue(session_id: str) -> None:
    """移除指定 session 的队列。"""
    logger.debug(f"[SSE] 移除队列 session={session_id}")
    _session_stream.pop(session_id, None)
    _session_stream_ts.pop(session_id, None)


def _sse_pack(event: str, data: Dict[str, Any]) -> str:
    """打包 SSE 消息格式。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def push_to_session(session_id: str, event: str, data: Dict[str, Any]) -> None:
    """
    通过 session_id 推送事件。
    若队列不存在则自动创建，避免客户端短暂断开导致后续推送全部丢失。
    """
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        stream_queue = create_sse_queue(session_id)
    stream_queue.put({"event": event, "data": data})


async def sse_generator(session_id: str, request=None):
    """
    SSE 生成器，供 FastAPI StreamingResponse 使用。

    request 参数保留以兼容调用方签名（不主动检测 is_disconnected()：
    沙箱/反代可能误报客户端断开导致答案丢失，改为依赖显式 __close__ 信号
    或真实连接异常结束；队列保留以支持重连）。
    """
    logger.debug(f"[SSE] 生成器启动 session={session_id}")
    stream_queue = get_sse_queue(session_id)
    if stream_queue is None:
        stream_queue = create_sse_queue(session_id)

    should_remove_queue = False
    loop = asyncio.get_running_loop()
    try:
        yield _sse_pack("ready", {})
        while True:
            try:
                msg = await loop.run_in_executor(None, stream_queue.get, True, 1.0)
            except queue.Empty:
                continue

            event = msg.get("event")
            data = msg.get("data")

            if event == "__close__":
                logger.debug(f"[SSE] 收到关闭信号 session={session_id}")
                should_remove_queue = True
                break

            yield _sse_pack(event, data)
    except (asyncio.CancelledError, ConnectionResetError, BrokenPipeError):
        logger.debug(f"[SSE] 客户端断开，保留队列 session={session_id}")
    except Exception as e:  # noqa: BLE001
        logger.debug(f"[SSE] 生成器异常 session={session_id}: {e}")
    finally:
        logger.debug(f"[SSE] 生成器结束 session={session_id}, remove={should_remove_queue}")
        if should_remove_queue:
            remove_sse_queue(session_id)
