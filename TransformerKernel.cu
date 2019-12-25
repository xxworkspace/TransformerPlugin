
#include <cuda_runtime.h>
#include <device_functions.h>
#include <sm_30_intrinsics.h>
#include "TransformerKernel.h"
#include <device_launch_parameters.h>

__inline__ __device__ float warp_reduce_sum(float value) {
  value += __shfl_down_sync(0xffffffff, value, 16, 32);
  value += __shfl_down_sync(0xffffffff, value, 8, 32);
  value += __shfl_down_sync(0xffffffff, value, 4, 32);
  value += __shfl_down_sync(0xffffffff, value, 2, 32);
  value += __shfl_down_sync(0xffffffff, value, 1, 32);

  return value;
}

__inline__ __device__ float block_allreduce_sum(float value) {
  __shared__ float tmp[32];
  if (threadIdx.x < 32)
    tmp[threadIdx.x] = 0.0;

  value = warp_reduce_sum(value);
  if (threadIdx.x % 32 == 0)
    tmp[threadIdx.x / 32] = value;
  __syncthreads();

  if (threadIdx.x < 32) {
    value = tmp[threadIdx.x];
    value = warp_reduce_sum(value);
    if (threadIdx.x == 0)
      tmp[0] = value;
  }
  __syncthreads();
  value = tmp[0];
  return value;
}

template <class T>
__global__ void cuLayerNormalization(const T *value,int col,
  const float *weight, const float *bias, T *output) {
  // id over col
  float tmpv = 0.0, gamma = 0.0, beta = 0.0;
  if (threadIdx.x < col) {
    tmpv = (float)__ldg(&value[blockIdx.x * col + threadIdx.x]);
    gamma = (float)__ldg(&weight[threadIdx.x]);
    beta = (float)__ldg(&bias[threadIdx.x]);
  }

  float sum = block_allreduce_sum(tmpv);
  float mean = sum / col;
  float diff = tmpv - mean;
  sum = block_allreduce_sum(diff*diff);
  float std = sqrtf(sum/col);

  tmpv = (tmpv - mean) / std * gamma + beta;
  if (threadIdx.x < col)
    output[threadIdx.x] = T(tmpv);
}

template<class T>
void layerNormalization(const T*inputs, const float*gamma, const float*beta, const int num, const int col, T* output, cudaStream_t stream) {
  dim3 dg(num);
  dim3 db((col + 31) / 32 * 32);
  ASSERT(col <= 512);
  cuLayerNormalization<T> << <dg, db, 0, stream >> > (inputs,col,gamma,beta,output);
}

template void layerNormalization<float>(const float*, const float*,
  const float*, const int, const int, float*, cudaStream_t);
template void layerNormalization<half>(const half*, const float*,
  const float*, const int, const int, half*, cudaStream_t);

template <class T> __global__ void cuMaskScore(T *score, int col) {
  const int tid = blockDim.x * blockIdx.x + threadIdx.x;
  const int id = tid % (col * col);
  T *tmp = score + tid;
  
  int row_id = id / col;
  int col_id = id % col;

  if (id < col*col && col_id > row_id)
    *tmp = T(-1.0e5);
}

template<class T>
void maskScore(T *inputs, const int num, const int col, cudaStream_t stream) {
  int size = num * col * col;
  dim3 dg(1);
  dim3 db(128);
  if (size < 128) {
    size = (size + 31) / 32 * 32;
    db.x = size;
  }
  else {
    dg.x = (size + 127) / 128;
  }

  cuMaskScore<T><<<dg, db, 0, stream>>>(inputs,col);
}

template void maskScore<float>(float*, const int, const int, cudaStream_t);
template void maskScore<half>(half*, const int, const int, cudaStream_t);

template <class T>
__global__ void cuFinalSlice(const T *input, const int seql,
  const int dim, T *output) {
  const T *tmpi = input + blockIdx.x * (seql * dim) + (seql - 1) * dim;
  T *tmpo = output + blockIdx.x * dim;

  for (int tid = threadIdx.x; tid < dim; tid += blockDim.x)
    tmpo[tid] = tmpi[tid];
}

template<class T>
void finalSlice(const T*input, const int batch,const int num, const int col, T* output, cudaStream_t stream) {
  dim3 dg(batch);
  ASSERT(col <= 512)
  dim3 db((col + 31) / 32 * 32);
  cuFinalSlice<T> << <dg, db, 0, stream >> > (input, num, col, output);
}

template void finalSlice<float>(const float *, const int,
  const int, const int, float *,cudaStream_t stream);
template void finalSlice<half>(const half *, const int,
  const int, const int, half *, cudaStream_t stream);