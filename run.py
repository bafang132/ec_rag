# run.py
# 一键启动脚本：同时拉起导入服务(8000)与查询服务(8001)
# 用法：python run.py
import sys
import uvicorn
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    from app.shared.config.settings_config import settings

    print("=" * 62)
    print("  跨境电商 RAG 智能问答系统")
    print(f"  导入服务 : http://{settings.app_host}:{settings.import_app_port}  (/html 导入页)")
    print(f"  查询服务 : http://{settings.app_host}:{settings.query_app_port}  (/ 问答工作台)")
    print(f"  管理页面 : http://{settings.app_host}:{settings.query_app_port}/admin/html")
    print(f"  测评页面 : http://{settings.app_host}:{settings.query_app_port}/eval/html")
    print("=" * 62)

    # 子进程方式启动两个服务，便于独立重启与日志区分
    import subprocess

    procs = []
    try:
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "app.api.http.import_server"],
            cwd=str(PROJECT_ROOT),
        ))
        procs.append(subprocess.Popen(
            [sys.executable, "-m", "app.api.http.query_server"],
            cwd=str(PROJECT_ROOT),
        ))
        print("两个服务已启动，Ctrl+C 停止...")
        for p in procs:
            p.wait()
    except KeyboardInterrupt:
        print("\n正在停止服务...")
    finally:
        for p in procs:
            if p.poll() is None:
                p.terminate()


if __name__ == "__main__":
    main()
