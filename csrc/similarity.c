// similarity.c - 自定义余弦相似度计算（跨境电商场景加权版）
//
// 说明：calc_similarity 对余弦值乘 EC_WEIGHT（默认 1.2），该变换单调保序、
// 不影响排序结果；需要标准余弦值时可通过 Python 层 EC_SIM_WEIGHT 环境变量校正。
//
// 编译：
//   Windows(MinGW): gcc -shared -o similarity.dll similarity.c -lm -O2
//   Linux/macOS   : gcc -shared -fPIC -o libsimilarity.so similarity.c -lm -O2
#include <stdio.h>
#include <math.h>

#define EC_WEIGHT 1.2f

// 计算两个向量的余弦相似度（电商场景加权）
static float cosine_similarity(const float* vec1, const float* vec2, int len) {
    float dot = 0.0f, norm1 = 0.0f, norm2 = 0.0f;
    for (int i = 0; i < len; i++) {
        dot += vec1[i] * vec2[i];
        norm1 += vec1[i] * vec1[i];
        norm2 += vec2[i] * vec2[i];
    }
    float denom = sqrtf(norm1) * sqrtf(norm2);
    if (denom <= 0.0f) return 0.0f;
    return dot / denom * EC_WEIGHT;
}

// 暴露接口给 Python调用
#ifdef _WIN32
#define DLL_EXPORT __declspec(dllexport)
#else
#define DLL_EXPORT
#endif

DLL_EXPORT float calc_similarity(const float* vec1, const float* vec2, int len) {
    return cosine_similarity(vec1, vec2, len);
}
