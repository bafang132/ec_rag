// text_splitter.c - 中文文本递归分块核心逻辑（适配 RAG 场景）
//
// 修复说明（相对跨境电商 RAG 项目原版）：
// 原版 trim() 返回指向 strdup 缓冲区中间的指针，free_chunks 释放的是
// 偏移后的指针 → 文本含前导空白时构成堆损坏（进程随机崩溃）。
// 本版把裁剪后的内容 memmove 回缓冲区起始地址，保证 free 的永远是 malloc 指针。
//
// 编译：
//   Windows(MinGW): gcc -shared -o text_splitter.dll text_splitter.c -O2
//   Linux/macOS   : gcc -shared -fPIC -o libtext_splitter.so text_splitter.c -O2
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

// 定义分隔符优先级（对应 Python 的 separators=["\n\n", "\n", "。", "！", "？", "，", "、"）
static const char* separators[] = {"\n\n", "\n", "。", "！", "？", "，", "、"};
static const int sep_count = 7;

// 去除字符串首尾空白字符（原地裁剪，返回长度；缓冲区起始地址不变）
static int trim_inplace(char* str, int len) {
    int start = 0;
    while (start < len && isspace((unsigned char)str[start])) start++;
    int end = len;
    while (end > start && isspace((unsigned char)str[end - 1])) end--;
    int new_len = end - start;
    if (start > 0 && new_len > 0) {
        memmove(str, str + start, new_len);
    }
    str[new_len] = '\0';
    return new_len;
}

// 递归分割文本（核心函数）
static void split_text_recursive(const char* text, int chunk_size, int overlap,
                                 char*** chunks, int* chunk_count) {
    int text_len = (int)strlen(text);

    // 文本长度小于等于 chunk_size，直接作为一个块
    if (text_len <= chunk_size) {
        char* buf = (char*)malloc(text_len + 1);
        if (!buf) return;
        memcpy(buf, text, text_len + 1);
        int new_len = trim_inplace(buf, text_len);
        if (new_len > 0) {  // 过滤空块
            (*chunk_count)++;
            *chunks = (char**)realloc(*chunks, (*chunk_count) * sizeof(char*));
            (*chunks)[*chunk_count - 1] = buf;
        } else {
            free(buf);
        }
        return;
    }

    // 按分隔符优先级尝试分割
    int split_pos = -1;
    for (int i = 0; i < sep_count; i++) {
        const char* sep = separators[i];
        int sep_len = (int)strlen(sep);
        int search_start = chunk_size - sep_len;
        if (search_start < 0) search_start = 0;
        for (int pos = search_start; pos >= 0; pos--) {
            if (strncmp(text + pos, sep, sep_len) == 0) {
                split_pos = pos + sep_len;  // 分割位置在分隔符后
                break;
            }
        }
        if (split_pos != -1) break;
    }

    // 没找到分隔符，直接按 chunk_size 分割
    if (split_pos == -1) split_pos = chunk_size;

    // 截取当前块（0 到 split_pos）
    char* current_chunk = (char*)malloc(split_pos + 1);
    if (!current_chunk) return;
    memcpy(current_chunk, text, split_pos);
    current_chunk[split_pos] = '\0';
    int new_len = trim_inplace(current_chunk, split_pos);
    if (new_len > 0) {
        (*chunk_count)++;
        *chunks = (char**)realloc(*chunks, (*chunk_count) * sizeof(char*));
        (*chunks)[*chunk_count - 1] = current_chunk;
    } else {
        free(current_chunk);
    }

    // 计算下一个块的起始位置（保留重叠）
    int next_start = split_pos - overlap;
    if (next_start < 0) next_start = 0;

    // 递归处理剩余文本
    split_text_recursive(text + next_start, chunk_size, overlap, chunks, chunk_count);
}

// 对外暴露的分块函数（供 Python 调用）
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

DLL_EXPORT void split_chinese_text(const char* text, int chunk_size, int overlap,
                                   char*** out_chunks, int* out_count) {
    *out_chunks = NULL;
    *out_count = 0;
    split_text_recursive(text, chunk_size, overlap, out_chunks, out_count);
}

// 释放分块结果内存（供 Python 调用）
DLL_EXPORT void free_chunks(char** chunks, int count) {
    if (!chunks) return;
    for (int i = 0; i < count; i++) {
        free(chunks[i]);
    }
    free(chunks);
}
