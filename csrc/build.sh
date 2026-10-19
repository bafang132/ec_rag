#!/usr/bin/env bash
# ============================================================
# C 加速库编译脚本（Linux / macOS / Git Bash）
# 依赖：gcc（sudo apt install gcc 或 xcode-select --install）
# 产出：csrc/prebuilt/libtext_splitter.so 与 libsimilarity.so
# ============================================================
set -e

cd "$(dirname "$0")"

if ! command -v gcc >/dev/null 2>&1; then
    echo "[ERROR] 未找到 gcc，请先安装（apt install gcc / xcode-select --install）"
    exit 1
fi

mkdir -p prebuilt

echo "[1/2] 编译 libtext_splitter.so ..."
gcc -shared -fPIC -o prebuilt/libtext_splitter.so text_splitter.c -O2

echo "[2/2] 编译 libsimilarity.so ..."
gcc -shared -fPIC -o prebuilt/libsimilarity.so similarity.c -lm -O2

echo ""
echo "[OK] 编译完成，动态库已输出到 csrc/prebuilt/"
echo "      Python 层加载时会执行自检，通过后自动启用 C 加速。"
