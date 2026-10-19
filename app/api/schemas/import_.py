"""
导入服务 API 数据模型。
"""
from pydantic import BaseModel


class UploadResponse(BaseModel):
    """多文件上传响应。"""
    code: int = 200
    message: str = ""
    task_ids: list[str] = []


class ImportStatusResponse(BaseModel):
    """导入任务状态响应。"""
    code: int = 200
    task_id: str = ""
    status: str = ""               # pending/processing/completed/failed
    done_list: list[str] = []      # 已完成节点（中文）
    running_list: list[str] = []   # 运行中节点（中文）
