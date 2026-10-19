"""
导入服务 HTTP 入口模块（8000）：文件上传、后台导入任务与进度查询。
"""
import shutil
import sys
import uuid
from datetime import datetime
from mimetypes import guess_type
from pathlib import Path

# 兼容直接以 `python import_server.py` 启动：提前把项目根加入模块搜索路径
if __package__ in (None, ""):
    bootstrap_root = Path(__file__).resolve().parents[3]
    if str(bootstrap_root) not in sys.path:
        sys.path.insert(0, str(bootstrap_root))

from fastapi import BackgroundTasks, FastAPI, File, UploadFile
from fastapi.responses import FileResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.http.auth import setup_auth
from app.api.schemas.import_ import ImportStatusResponse, UploadResponse
from app.process.import_.agent.main_graph import kb_import_app
from app.process.import_.agent.state import get_default_state
from app.rag.import_.version_service import file_hash as compute_file_hash
from app.rag.import_.version_service import is_unchanged
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.utils.task_utils import (
    add_done_task,
    add_running_task,
    get_done_task_list,
    get_running_task_list,
    get_task_status,
    update_task_status,
)

app = FastAPI(
    title=settings.import_app_name,
    description="跨境电商知识库导入服务：文件上传、导入执行与状态查询。",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(settings.cors_origins) or ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# 可选 API Key 鉴权 + 多租户隔离（默认关闭，关闭时行为与旧版一致）
setup_auth(app)


@app.get("/html")
def import_html():
    """返回导入演示页面。"""
    html_path = PROJECT_ROOT / "app" / "process" / "import_" / "page" / "import.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/health")
def health_check():
    """服务存活探测（前端状态药丸轮询）。"""
    return {"status": "ok", "service": settings.import_app_name}


def run_graph_task(task_id: str, local_dir: str, local_file_path: str,
                   item_name: str = "", file_hash: str = ""):
    """
    LangGraph 全流程后台任务。

    流式执行图节点并实时更新任务状态（pending → processing → completed/failed），
    每个节点完成后加入 done_list，供前端轮询查看进度。
    """
    try:
        update_task_status(task_id, "processing")
        logger.info(f"[{task_id}] 开始执行导入全流程：{local_file_path}")

        init_state = get_default_state()
        init_state["task_id"] = task_id
        init_state["local_dir"] = local_dir
        init_state["local_file_path"] = local_file_path
        init_state["item_name"] = item_name
        init_state["file_hash"] = file_hash

        for event in kb_import_app.stream(init_state):
            for node_name in event:
                logger.info(f"[{task_id}] 节点执行完成：{node_name}")

        update_task_status(task_id, "completed")
        logger.info(f"[{task_id}] 导入全流程执行完毕")
    except Exception as exc:  # noqa: BLE001
        update_task_status(task_id, "failed")
        logger.error(f"[{task_id}] 导入流程失败：{exc}", exc_info=True)


@app.post("/upload", summary="文件上传接口",
          description="支持多文件批量上传（PDF/MD/DOCX/TXT/HTML/XLSX/PPTX/CSV），自动触发导入全流程")
async def upload_files(background_tasks: BackgroundTasks, files: list[UploadFile] = File(...)):
    """
    多文件上传：本地保存 → 每文件独立 TaskID → 后台 LangGraph 导入。

    文件按 output/YYYYMMDD/<task_id>/ 分层保存，避免重名冲突。
    """
    today_str = datetime.now().strftime("%Y%m%d")
    date_based_root_dir: Path = PROJECT_ROOT / "output" / today_str
    task_ids: list[str] = []

    for file in files:
        task_id = str(uuid.uuid4())
        task_ids.append(task_id)
        logger.info(f"[{task_id}] 开始处理上传文件：{file.filename}")

        add_running_task(task_id, "upload_file")

        task_local_dir: Path = date_based_root_dir / task_id
        task_local_dir.mkdir(parents=True, exist_ok=True)
        local_file_abs_path: Path = task_local_dir / file.filename

        with local_file_abs_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"[{task_id}] 文件已保存：{local_file_abs_path}")

        add_done_task(task_id, "upload_file")

        # item_name 默认取文件名（去扩展名），支持同主体覆盖更新
        item_name = Path(file.filename).stem
        content_hash = compute_file_hash(str(local_file_abs_path))

        # 增量更新：内容指纹与上次一致则跳过重索引，直接标记完成
        if is_unchanged(item_name, content_hash):
            add_running_task(task_id, "增量检查")
            add_done_task(task_id, "内容未变化-跳过重索引")
            update_task_status(task_id, "completed")
            logger.info(f"[{task_id}] 主体[{item_name}] 内容未变化，跳过重索引（增量更新）")
        else:
            background_tasks.add_task(
                run_graph_task, task_id, str(task_local_dir), str(local_file_abs_path), item_name, content_hash
            )
            logger.info(f"[{task_id}] 导入任务已加入后台队列")

    return UploadResponse(
        code=200,
        message=f"Files uploaded successfully, total: {len(files)}",
        task_ids=task_ids,
    )


@app.get("/status/{task_id}", summary="任务状态查询",
         description="根据 TaskID 查询单个文件的处理进度与全局状态",
         response_model=ImportStatusResponse)
async def get_task_progress(task_id: str):
    """查询导入任务进度（前端每秒轮询）。"""
    status = get_task_status(task_id)
    done_list = get_done_task_list(task_id)
    running_list = get_running_task_list(task_id)
    return ImportStatusResponse(
        code=200,
        task_id=task_id,
        status=status,
        done_list=done_list,
        running_list=running_list,
    )


@app.post("/reindex/{item_name}", summary="重新索引指定文档",
          description="删除指定主体的旧切片（重新上传文件完成增量更新）")
async def reindex_item(item_name: str):
    """增量更新第一步：清除旧切片，引导重新上传。"""
    from app.rag.admin.service import delete_item

    result = delete_item(item_name)
    return {
        "code": 200 if result.get("success") else 404,
        "message": f"已清除主体[{item_name}]的旧切片，请重新上传文件以完成增量更新",
        **result,
    }


@app.get("/api/import/versions/{item_name}", summary="查询文档版本历史",
         description="返回该主体的导入版本记录（指纹 / 分片数 / 时间戳），支持增量更新判断")
async def import_versions(item_name: str):
    """文档版本管理：查看某主体的历史导入版本。"""
    from app.rag.import_.version_service import list_versions

    versions = list_versions(item_name)
    return {
        "code": 200,
        "item_name": item_name,
        "count": len(versions),
        "versions": versions,
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.import_app_port)
