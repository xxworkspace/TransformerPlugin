
#include "TransformerKernel.h"

template<class T>
void layerNormalization(const T*inputs, const float*gamma, const float*beta, const int num, const int col, T* output, cudaStream_t stream) {

}

template void layerNormalization<float>(const float*, const float*,
  const float*, const int, const int, float*, cudaStream_t);
template void layerNormalization<half>(const half*, const float*,
  const float*, const int, const int, half*, cudaStream_t);

template<class T>
void MaskScore(T *inputs, const int num, const int col, cudaStream_t stream) {

}

template void MaskScore<float>(float*, const int, const int, cudaStream_t);
template void MaskScore<half>(half*, const int, const int, cudaStream_t);

template<class T>
void finalSlice(const T*input, const int batch,const int num, const int col, T* output, cudaStream_t stream) {

}

template void finalSlice<float>(const float *, const int,
  const int, const int, float *,cudaStream_t stream);
template void finalSlice<half>(const half *, const int,
  const int, const int, half *, cudaStream_t stream);