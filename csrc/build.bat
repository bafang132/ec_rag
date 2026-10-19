@echo off
REM ============================================================
REM  C 加速库编译脚本（Windows / MinGW）
REM  依赖：MinGW-w64（https://www.mingw-w64.org/），gcc 已在 PATH
REM  产出：csrc\prebuilt\text_splitter.dll 与 similarity.dll
REM ============================================================
setlocal

where gcc >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 未找到 gcc，请先安装 MinGW-w64 并将其 bin 目录加入 PATH
    echo         下载地址：https://www.mingw-w64.org/downloads/
    exit /b 1
)

if not exist prebuilt mkdir prebuilt

echo [1/2] 编译 text_splitter.dll ...
gcc -shared -o prebuilt\text_splitter.dll text_splitter.c -O2
if errorlevel 1 goto :failed

echo [2/2] 编译 similarity.dll ...
gcc -shared -o prebuilt\similarity.dll similarity.c -lm -O2
if errorlevel 1 goto :failed

echo.
echo [OK] 编译完成，动态库已输出到 csrc\prebuilt\
echo       Python 层加载时会执行自检，通过后自动启用 C 加速。
exit /b 0

:failed
echo [ERROR] 编译失败，请检查 gcc 安装与源码
exit /b 1
