"""
查询服务 HTTP 入口模块（8001）：智能问答（SSE 流式）、会话管理、知识库管理、专业测评。

服务启动时会 bootstrap sys.path（项目根 + evaluation/），保证 /api/eval/* 路由可导入测评模块。
"""
import sys
import uuid
from contextlib import asynccontextmanager
from mimetypes import guess_type
from pathlib import Path

# ---------------- sys.path bootstrap ----------------
# 确保「项目根」与「evaluation/」都在 sys.path，使 evaluation 包及其内部
# `from metrics import ...` 等导入均可解析（uvicorn 启动时 __package__ 非空）。
_bootstrap_root = Path(__file__).resolve().parents[3]
_bootstrap_eval = _bootstrap_root / "evaluation"
for _p in (_bootstrap_root, _bootstrap_eval):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from fastapi import BackgroundTasks, FastAPI, Query, Request, UploadFile, File
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

from app.api.http.auth import setup_auth
from app.api.schemas.ec import (
    AdminChunkItem,
    AdminGetChunksResponse,
    AdminItemInfo,
    AdminListItemsResponse,
    HistoryItem,
    HistoryResponse,
    QACompatRequest,
    QueryRequest,
    QueryResponse,
    SessionRenameRequest,
)
from app.infra.persistence import history_repository
from app.infra.vectorstore import vectorstore_gateway
from app.process.query.agent.main_graph import query_app as query_graph_app
from app.process.query.agent.state import create_query_default_state
from app.rag.admin.service import (
    delete_item as admin_delete_item,
    get_chunks as admin_get_chunks,
    list_items as admin_list_items,
)
from app.rag.session.service import (
    create_session as session_create,
    list_all_sessions as session_list_all,
    remove_session as session_remove,
    rename_session_title as session_rename,
)
from app.shared.config.settings_config import settings
from app.shared.runtime.logger import PROJECT_ROOT, logger
from app.shared.utils.sse_utils import (
    SSEEvent,
    create_sse_queue,
    push_to_session,
    sse_generator,
)
from app.shared.utils.task_utils import (
    TASK_STATUS_COMPLETED,
    TASK_STATUS_FAILED,
    TASK_STATUS_PROCESSING,
    clear_task,
    get_done_task_list,
    get_task_result,
    update_task_status,
)

app = FastAPI(
    title=settings.query_app_name,
    description="跨境电商 RAG 智能问答服务：问答对话、会话管理、知识库管理与专业测评。",
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


# ======================== 生命周期：预热向量库 ========================

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("服务启动中，预热向量库...")
    try:
        vectorstore_gateway.ensure_ready()
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"向量库预热失败（不影响服务启动）：{exc}")
    yield
    logger.info("服务关闭")


app.router.lifespan_context = lifespan


# ======================== 通用工具 ========================

def new_session_id() -> str:
    """生成新的会话 ID。"""
    return str(uuid.uuid4())


def invoke_query(session_id: str, query: str, is_stream: bool,
                 use_web_search: bool = False, role: str = "seller") -> dict:
    """调用查询主图并维护统一任务状态。"""
    clear_task(session_id)
    update_task_status(session_id, TASK_STATUS_PROCESSING, is_stream)
    initial_state = create_query_default_state(
        session_id=session_id,
        original_query=query,
        is_stream=is_stream,
        use_web_search=use_web_search,
        role=role,
    )
    state = query_graph_app.invoke(initial_state)
    update_task_status(session_id, TASK_STATUS_COMPLETED, is_stream)
    return state


def run_stream_query_background(session_id: str, query: str,
                                use_web_search: bool = False, role: str = "seller") -> None:
    """后台执行流式查询任务，经 SSE 推送增量与最终结果。"""
    try:
        state = invoke_query(session_id, query, is_stream=True,
                             use_web_search=use_web_search, role=role)
        answer = get_task_result(session_id, "answer") or state.get("answer", "")
        push_to_session(session_id, SSEEvent.FINAL, {
            "answer": answer,
            "status": "completed",
            "image_urls": state.get("image_urls", []),
            "data_source": "web" if use_web_search else "kb",
        })
        push_to_session(session_id, SSEEvent.CLOSE, {})
    except Exception as exc:  # noqa: BLE001
        logger.exception(f"流式查询执行失败，session_id={session_id}, error={exc}")
        update_task_status(session_id, TASK_STATUS_FAILED, True)
        push_to_session(session_id, SSEEvent.ERROR,
                        {"message": "查询执行失败，请检查日志或稍后重试。"})
        push_to_session(session_id, SSEEvent.CLOSE, {})


def start_stream_query(background_tasks: BackgroundTasks, query: str,
                       session_id: str | None = None, use_web_search: bool = False,
                       role: str = "seller") -> QueryResponse:
    """启动流式查询任务。"""
    final_session_id = session_id or new_session_id()
    create_sse_queue(final_session_id)
    background_tasks.add_task(run_stream_query_background, final_session_id, query,
                              use_web_search, role)
    return QueryResponse(message="结果正在处理中...", session_id=final_session_id)


def execute_query(query: str, session_id: str | None = None,
                  use_web_search: bool = False, role: str = "seller") -> QueryResponse:
    """非流式执行查询。"""
    final_session_id = session_id or new_session_id()
    state = invoke_query(final_session_id, query, is_stream=False,
                         use_web_search=use_web_search, role=role)
    answer = get_task_result(final_session_id, "answer") or state.get("answer", "")
    return QueryResponse(
        message="处理完成！",
        session_id=final_session_id,
        answer=answer,
        done_list=get_done_task_list(final_session_id),
        image_urls=state.get("image_urls", []),
    )


# ======================== 健康检查 & 导航 ========================

@app.get("/health")
def health():
    """服务健康检查。"""
    return {
        "ok": True,
        "app": settings.query_app_name,
        "env": settings.app_env,
        "module": "ec-rag",
        "vector_backend": vectorstore_gateway.backend,
    }


@app.get("/")
def index():
    """统一工作台页面（智能问答 + 知识库导入入口）。"""
    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "chat.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/api")
def api_info():
    """纯 JSON 服务导航（机器可读）。"""
    return {
        "message": settings.query_app_name,
        "docs": "/docs",
        "openapi": "/openapi.json",
        "workspace": "/",
        "routes": {
            "html": "/html",
            "query": "/query",
            "qa": "/qa",
            "stream": "/stream/{session_id}",
            "history": "/history/{session_id}",
            "sessions": "/sessions",
            "admin_items": "/admin/items",
            "eval": "/api/eval/*",
        },
    }


# ======================== 问答对话接口 ========================

@app.get("/html")
def query_html():
    """返回问答工作台页面。"""
    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "chat.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest, background_tasks: BackgroundTasks):
    """
    统一问答入口：普通问答 + 流式问答（is_stream=true 走 SSE）。

    非流式分支经线程池卸载，避免阻塞事件循环冻结其它并发请求。
    """
    if request.is_stream:
        return start_stream_query(
            background_tasks=background_tasks,
            query=request.query,
            session_id=request.session_id,
            use_web_search=request.use_web_search,
            role=request.role,
        )
    return await run_in_threadpool(
        execute_query,
        query=request.query,
        session_id=request.session_id,
        use_web_search=request.use_web_search,
        role=request.role,
    )


@app.post("/qa", summary="简化问答接口（兼容跨境电商 RAG V1）")
async def qa_api(request: QACompatRequest):
    """简化问答接口：接收 {question}（兼容 V1 契约），返回答案文本。"""
    try:
        response = await run_in_threadpool(
            execute_query, query=request.question, role=request.role
        )
        return {"code": 200, "question": request.question, "answer": response.answer}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"问答接口失败：{exc}", exc_info=True)
        return JSONResponse(status_code=500, content={"code": 500, "error": str(exc)})


@app.get("/stream/{session_id}")
async def stream_query_result(session_id: str, request: Request):
    """建立指定会话的 SSE 结果流。"""
    return StreamingResponse(
        sse_generator(session_id, request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/history/{session_id}", response_model=HistoryResponse)
def history(session_id: str, limit: int = 10):
    """查询会话历史记录。"""
    records = history_repository.list_recent(session_id, limit=limit)
    items = [
        HistoryItem(
            _id=str(record.get("_id")) if record.get("_id") is not None else "",
            session_id=record.get("session_id", ""),
            role=record.get("role", ""),
            text=record.get("text", ""),
            rewritten_query=record.get("rewritten_query", ""),
            item_names=record.get("item_names") or [],
            image_urls=record.get("image_urls") or [],
            ts=record.get("ts"),
        )
        for record in records
    ]
    return HistoryResponse(session_id=session_id, items=items)


@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    """清空指定会话的聊天记录。"""
    deleted = history_repository.clear_session(session_id)
    return {"message": f"删除会话[{session_id}]聊天记录成功", "deleted_count": deleted}


# ======================== 会话管理 ========================

@app.get("/sessions", summary="列出所有会话")
def list_sessions_route(limit: int = Query(default=50, ge=1, le=200)):
    """返回会话摘要列表（多会话侧边栏）。"""
    try:
        data = session_list_all(limit=limit)
        return {"code": 200, "message": "查询成功", "data": data}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"列出会话失败：{exc}", exc_info=True)
        return {"code": 500, "message": str(exc), "data": []}


@app.post("/sessions", summary="创建新会话")
def create_session_route():
    """创建新会话（必须落库，否则空会话不出现在列表）。"""
    session_id = new_session_id()
    ok = session_create(session_id)
    return {
        "code": 200 if ok else 500,
        "message": "创建成功" if ok else "创建失败",
        "session_id": session_id if ok else "",
    }


@app.delete("/sessions/{session_id}", summary="删除会话")
def delete_session_route(session_id: str):
    """删除会话及其全部聊天记录。"""
    deleted = session_remove(session_id)
    return {"code": 200, "message": "删除成功", "deleted_count": deleted}


@app.patch("/sessions/{session_id}", summary="重命名会话")
def rename_session_route(session_id: str, body: SessionRenameRequest):
    """设置会话自定义标题。"""
    ok = session_rename(session_id, body.title)
    return {"code": 200 if ok else 400, "message": "更新成功" if ok else "更新失败"}


# ======================== 知识库管理 ========================

@app.get("/admin/html")
def admin_html():
    """返回知识库管理页面。"""
    html_path = PROJECT_ROOT / "app" / "process" / "query" / "page" / "admin.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/admin/items", response_model=AdminListItemsResponse)
def admin_list_items_route(limit: int = Query(default=10000, ge=1, le=100000)):
    """列出知识库所有主体（按平台/市场/类目聚合元信息）。"""
    data = admin_list_items(limit=limit)
    items = [AdminItemInfo(**item) for item in data]
    return AdminListItemsResponse(data=items)


@app.delete("/admin/items/{item_name}")
def admin_delete_item_route(item_name: str):
    """删除指定主体及其所有切片。"""
    result = admin_delete_item(item_name)
    code = 200 if result.get("success") else 404
    return {"code": code, **result}


@app.get("/admin/items/{item_name}/chunks", response_model=AdminGetChunksResponse)
def admin_get_chunks_route(item_name: str, limit: int = Query(default=20, ge=1, le=200)):
    """查看指定主体的切片详情。"""
    chunks = admin_get_chunks(item_name, limit=limit)
    return AdminGetChunksResponse(
        item_name=item_name,
        chunks=[AdminChunkItem(**c) for c in chunks],
        total=len(chunks),
    )


# ======================== 平台规则模板 ========================

@app.get("/api/platform-templates", summary="平台规则模板")
def platform_templates():
    """返回各跨境电商平台的规则重点 / 合规关注 / 推荐文档类型（前端导入引导用）。"""
    from app.rag.import_.platform_templates import PLATFORM_TEMPLATES

    return {"code": 200, "platforms": list(PLATFORM_TEMPLATES.keys()),
            "templates": PLATFORM_TEMPLATES}


# ======================== 本地静态文件（图片） ========================

@app.get("/api/files/{file_path:path}")
def serve_local_file(file_path: str):
    """本地静态文件服务（MinIO 未配置时的文档图片访问）。"""
    from app.infra.object_storage import object_storage_gateway

    local_path = object_storage_gateway.resolve_local_path(file_path)
    if local_path and local_path.exists():
        return FileResponse(path=str(local_path), media_type=guess_type(local_path.name)[0])
    return JSONResponse(status_code=404, content={"error": "file not found"})


# ======================== 专业测评 ========================

_EVAL_TASKS: dict = {}
_EVAL_LOCK = __import__("threading").Lock()


def _ensure_eval_path():
    """防御性注入：确保 evaluation 包可导入。"""
    for p in (_bootstrap_root, _bootstrap_eval):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))


def _eval_worker(task_id: str, mode: str, use_llm: bool, n: int = 0,
                 compare: bool = False, third_party: bool = False):
    """后台线程运行评估（offline 模式真实调用 query_app 图）。"""
    import traceback as _tb

    try:
        _ensure_eval_path()
        from evaluation.runner import run as _run_eval

        def _progress(done, total, current):
            with _EVAL_LOCK:
                _EVAL_TASKS[task_id]["progress"] = {"done": done, "total": total, "current": current}

        report = _run_eval(mode=mode, use_llm=use_llm,
                           n=(n if n > 0 else None), progress_cb=_progress,
                           compare=compare, third_party=third_party)
        with _EVAL_LOCK:
            total = report.get("meta", {}).get("n_cases", 0)
            _EVAL_TASKS[task_id]["progress"] = {"done": total, "total": total, "current": ""}
            _EVAL_TASKS[task_id]["report"] = report
            _EVAL_TASKS[task_id]["status"] = "done"
    except Exception as exc:  # noqa: BLE001
        with _EVAL_LOCK:
            _EVAL_TASKS[task_id]["status"] = "error"
            _EVAL_TASKS[task_id]["error"] = repr(exc)
            _EVAL_TASKS[task_id]["trace"] = _tb.format_exc()


@app.get("/eval/html")
def eval_html():
    """返回专业测评页面。"""
    html_path = PROJECT_ROOT / "app" / "process" / "eval" / "page" / "eval.html"
    return FileResponse(path=html_path, media_type=guess_type(html_path.name)[0])


@app.get("/api/eval/testset")
def eval_testset():
    """返回测试集概览（条数 + 前几条样例）。"""
    try:
        _ensure_eval_path()
        from evaluation.runner import load_testset

        cases = load_testset()
        samples = [
            {"id": c.get("id"), "query": c.get("query"), "role": c.get("role", "seller")}
            for c in cases[:6]
        ]
        return JSONResponse({"count": len(cases), "samples": samples})
    except Exception as exc:  # noqa: BLE001
        return JSONResponse({"count": 0, "samples": [], "error": repr(exc)})


@app.post("/api/eval/run")
def eval_run(
    use_llm: bool = Query(default=True),
    mode: str = Query(default="offline"),
    n: int = Query(default=0),
    compare: bool = Query(default=False),
    third_party: bool = Query(default=False),
):
    """启动一次测评（后台线程），返回 task_id 供前端轮询。

    third_party=true 时，在自研评测之外额外调用 RAGAS / TruLens（需安装可选依赖）。
    """
    task_id = uuid.uuid4().hex
    with _EVAL_LOCK:
        _EVAL_TASKS[task_id] = {"status": "running", "report": None, "error": None,
                                "trace": None, "progress": {"done": 0, "total": 0, "current": ""}}
    import threading

    t = threading.Thread(target=_eval_worker,
                         args=(task_id, mode, use_llm, n, compare, third_party), daemon=True)
    t.start()
    return JSONResponse({"task_id": task_id, "status": "running"})


@app.get("/api/eval/status")
def eval_status(task_id: str = Query(...)):
    """轮询测评任务状态与结果。"""
    with _EVAL_LOCK:
        task = _EVAL_TASKS.get(task_id)
    if not task:
        return JSONResponse({"status": "not_found"}, status_code=404)
    return JSONResponse(task)


# ======================== 语音交互（可选） ========================

@app.post("/voice/transcribe", summary="语音转文字（STT）")
async def voice_transcribe(file: UploadFile = File(...)):
    """接收音频文件，返回识别文本。"""
    try:
        from app.rag.voice.service import transcribe_from_upload

        text = transcribe_from_upload(file)
        return {"code": 200, "text": text}
    except Exception as exc:  # noqa: BLE001
        logger.error(f"语音转文字失败：{exc}")
        return {"code": 400, "error": str(exc)}


@app.post("/tts", summary="文字转语音（TTS）")
async def text_to_speech_post(body: dict):
    """接收 {text}，返回合成音频流（WAV）。"""
    text = (body or {}).get("text", "")
    if not text or not str(text).strip():
        return JSONResponse(status_code=400, content={"error": "empty text"})
    try:
        from app.rag.voice.service import text_to_speech

        audio_data = text_to_speech(str(text))
        return StreamingResponse(iter([audio_data]), media_type="audio/wav")
    except Exception as exc:  # noqa: BLE001
        return JSONResponse(status_code=500, content={"error": str(exc)})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.app_host, port=settings.query_app_port)
