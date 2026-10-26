#pragma once

#include "common.cuh"

void ggml_cuda_cpy_f16_planar3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_planar4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso3(const char * src, char * dst, int64_t ne, cudaStream_t stream);
void ggml_cuda_cpy_f16_iso4(const char * src, char * dst, int64_t ne, cudaStream_t stream);
