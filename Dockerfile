# ============================================================
# 跨境电商 RAG 智能问答系统 - 生产镜像
# 单容器同时承载导入服务(8000)与查询服务(8001)，一键部署。
# ============================================================
FROM python:3.11-slim

# 系统依赖：gcc 用于编译 C 加速库（不编译也能跑，编译后更快）
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# 先装依赖（利用层缓存）
COPY requirements.txt pyproject.toml ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# 拷贝源码
COPY . .

# 编译 C 加速库（失败不影响运行，系统会自动降级纯 Python）
RUN cd csrc && (bash build.sh || echo "[WARN] C 库编译失败，将使用纯 Python 实现") && cd ..

# 数据持久化卷
VOLUME ["/app/data", "/app/output", "/app/logs"]

EXPOSE 8000 8001

# 一键启动（导入 + 查询双服务）
CMD ["python", "run.py"]
