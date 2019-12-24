#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>

template<class T>
void layerNormalization(const T*, const float*, const float*, const int, const int, T*, cudaStream_t);

template<class T>
void MaskScore(T *, const int, const int, cudaStream_t);

template<class T>
void finalSlice(const T*, const int, const int, const int, T*, cudaStream_t);
