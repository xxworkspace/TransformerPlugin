
#include "TransformerKernel.h"

template<class T>
void Embedding(const T* embedding, const int* input, const int num, const int col, T* output, cudaStream_t stream) {

}

template void Embedding<float>(const float*, const int *, const int, const int, float*, cudaStream_t);
template void Embedding<half>(const half*, const int *, const int, const int, half*, cudaStream_t);

template<class T>
void LayerNormalization(const T*inputs, const float*gamma, const float*beta, const int num, const int col, T* output, cudaStream_t stream) {

}

template void LayerNormalization<float>(const float*, const float*,
  const float*, const int, const int, float*, cudaStream_t);
template void LayerNormalization<half>(const half*, const float*,
  const float*, const int, const int, half*, cudaStream_t);

template<class T>
void AddBias(T* input, const T*bias, const int num, const int col, cudaStream_t stream) {

}

template void AddBias<float>(float*, const float*, const int, const int, cudaStream_t);
template void AddBias<half>(half*, const half*, const int, const int, cudaStream_t);

template<class T>
void MaskScore(T *inputs, const int num, const int col, cudaStream_t stream) {

}

template void MaskScore<float>(float*, const int, const int, cudaStream_t);
template void MaskScore<half>(half*, const int, const int, cudaStream_t);

template<class T>
void FinalSlice(const T*input, const int batch, const int stride, const int num, const int col, T* output, cudaStream_t stream) {

}

template void FinalSlice<float>(const float *, const int , const int,
  const int, const int, float *,cudaStream_t stream);
template void FinalSlice<half>(const half *, const int, const int,
  const int, const int, half *, cudaStream_t stream);