"""
任务追踪工具模块：内存态记录导入/查询任务的节点进度与状态（单进程）。
"""
from typing import Dict, List

from app.shared.utils.sse_utils import push_to_session

# key: task_id -> value: 节点名列表
_tasks_running_list: Dict[str, List[str]] = {}
_tasks_done_list: Dict[str, List[str]] = {}
# key: task_id -> value: 状态字符串
_tasks_status: Dict[str, str] = {}
# key: task_id -> value: 任务结果（如 query 的 answer）
_tasks_result: Dict[str, Dict[str, str]] = {}

TASK_STATUS_PENDING = "pending"
TASK_STATUS_PROCESSING = "processing"
TASK_STATUS_COMPLETED = "completed"
TASK_STATUS_FAILED = "failed"

# 节点名 -> 中文名映射（前端展示）
_NODE_NAME_TO_CN: Dict[str, str] = {
    # ---- 导入链 ----
    "upload_file": "开始上传文件",
    "node_entry": "检查文件",
    "node_doc_to_md": "文档转Markdown",
    "node_md_img": "Markdown图片处理",
    "node_document_split": "文档切分",
    "node_metadata_extract": "元数据抽取",
    "node_embedding": "向量生成",
    "node_index": "导入向量库",
    # ---- 查询链 ----
    "node_ec_intent": "意图识别",
    "node_search": "混合检索",
    "node_rrf": "RRF融合",
    "node_rerank": "重排序",
    "node_answer_output": "生成答案",
    "END": "处理完成",
}


def _ensure_task(task_id: str) -> None:
    """确保 task_id 对应的数据结构已初始化。"""
    _tasks_running_list.setdefault(task_id, [])
    _tasks_done_list.setdefault(task_id, [])
    _tasks_result.setdefault(task_id, {})


def _to_cn(node_name: str) -> str:
    """节点名转中文展示名；无映射返回原名。"""
    return _NODE_NAME_TO_CN.get(node_name, node_name)


def add_running_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """添加「正在运行」的节点任务。"""
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    if node_name not in running:
        running.append(node_name)
    if is_stream:
        task_push_queue(task_id)


def add_done_task(task_id: str, node_name: str, is_stream: bool = False) -> None:
    """添加「已完成」的节点任务（同时从运行中列表移除同名节点）。"""
    _ensure_task(task_id)
    running = _tasks_running_list[task_id]
    _tasks_running_list[task_id] = [n for n in running if n != node_name]
    done = _tasks_done_list[task_id]
    if node_name not in done:
        done.append(node_name)
    if is_stream:
        task_push_queue(task_id)


def set_task_result(task_id: str, key: str, value: str) -> None:
    """存储任务结果字段（如 answer / error）。"""
    _ensure_task(task_id)
    _tasks_result[task_id][key] = value


def get_task_result(task_id: str, key: str, default: str = "") -> str:
    """获取任务结果字段。"""
    _ensure_task(task_id)
    return _tasks_result.get(task_id, {}).get(key, default)


def get_task_status(task_id: str) -> str:
    """获取当前任务状态。"""
    return _tasks_status.get(task_id, "")


def get_done_task_list(task_id: str) -> List[str]:
    """获取已完成节点列表（中文展示）。"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_done_list.get(task_id, [])]


def get_running_task_list(task_id: str) -> List[str]:
    """获取正在运行节点列表（中文展示）。"""
    _ensure_task(task_id)
    return [_to_cn(n) for n in _tasks_running_list.get(task_id, [])]


def update_task_status(task_id: str, status_name: str, push_queue: bool = False) -> None:
    """更新任务状态。"""
    _tasks_status[task_id] = status_name
    if push_queue:
        task_push_queue(task_id)


def task_push_queue(task_id: str) -> None:
    """把当前任务进度推送到 SSE 队列（流式场景）。"""
    push_to_session(task_id, "progress", {
        "status": get_task_status(task_id),
        "done_list": get_done_task_list(task_id),
        "running_list": get_running_task_list(task_id),
    })


def clear_task(task_id: str) -> None:
    """清理任务全部追踪数据。"""
    _tasks_running_list.pop(task_id, None)
    _tasks_done_list.pop(task_id, None)
    _tasks_status.pop(task_id, None)
    _tasks_result.pop(task_id, None)
