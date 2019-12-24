/*
 * Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "selfAttentionPlugin.h"
#include "TransformerKernel.h"
#include "cublas_v2.h"
#include <cstring>
#include <cudnn.h>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::SelfAttention;
using nvinfer1::plugin::SelfAttentionPluginCreator;

namespace
{
  const char* SELF_ATTENTION_PLUGIN_VERSION{ "1" };
  const char* SELF_ATTENTION_PLUGIN_NAME{ "SelfAttention_TRT" };
} // namespace

PluginFieldCollection SelfAttentionPluginCreator::mFC{};
std::vector<PluginField> SelfAttentionPluginCreator::mPluginAttributes;

SelfAttention::SelfAttention(const int nhead,const int nfeat,int mask) {
  n_Head = nhead;
  n_Feat = nfeat;
  Mask = mask;

  for (int i = 0; i < 2; ++i) {
    CHECK(cudaMallocHost(&A[i], sizeof(void*) * 1024));
    CHECK(cudaMallocHost(&B[i], sizeof(void*) * 1024));
    CHECK(cudaMallocHost(&C[i], sizeof(void*) * 1024));
  }
  CHECK(cudaMalloc(&gA, sizeof(void*) * 1024));
  CHECK(cudaMalloc(&gB, sizeof(void*) * 1024));
  CHECK(cudaMalloc(&gC, sizeof(void*) * 1024));

  //cudnn compute
  CHECK(cudnnCreate(&dnnHandle));
  CHECK(cublasCreate(&blasHandle));
  CHECK(cudnnCreateTensorDescriptor(&xdes));
}

SelfAttention::SelfAttention(const SelfAttention& other) {
  n_Head = other.n_Head;
  n_Feat = other.n_Feat;
  Mask = other.Mask;
  ctype = other.ctype;
  mNameSpace = other.mNameSpace;

  for (int i = 0; i < 2; ++i) {
    CHECK(cudaMallocHost(&A[i], sizeof(void*) * 1024));
    CHECK(cudaMallocHost(&B[i], sizeof(void*) * 1024));
    CHECK(cudaMallocHost(&C[i], sizeof(void*) * 1024));
  }
  CHECK(cudaMalloc(&gA, sizeof(void*) * 1024));
  CHECK(cudaMalloc(&gB, sizeof(void*) * 1024));
  CHECK(cudaMalloc(&gC, sizeof(void*) * 1024));

  //cudnn compute
  CHECK(cudnnCreate(&dnnHandle));
  CHECK(cublasCreate(&blasHandle));
  CHECK(cudnnCreateTensorDescriptor(&xdes));
}

SelfAttention::SelfAttention(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  n_Head = read<int>(ptr);
  n_Feat = read<int>(ptr);
  Mask = read<int>(ptr);
  ctype = (DataType)read<int>(ptr);

  for (int i = 0; i < 2; ++i) {
    CHECK(cudaMallocHost(&A[i], sizeof(void*) * 1024));
    CHECK(cudaMallocHost(&B[i], sizeof(void*) * 1024));
    CHECK(cudaMallocHost(&C[i], sizeof(void*) * 1024));
  }
  CHECK(cudaMalloc(&gA, sizeof(void*) * 1024));
  CHECK(cudaMalloc(&gB, sizeof(void*) * 1024));
  CHECK(cudaMalloc(&gC, sizeof(void*) * 1024));

  //cudnn compute
  CHECK(cudnnCreate(&dnnHandle));
  CHECK(cublasCreate(&blasHandle));
  CHECK(cudnnCreateTensorDescriptor(&xdes));
}

IPluginV2DynamicExt* SelfAttention::clone()const {
  return (IPluginV2DynamicExt*)new SelfAttention(*this);
}

DimsExprs SelfAttention::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs* inputs,
  int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 3);
  return inputs[0];
}

bool SelfAttention::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc* inOut,
  int nbInputs, int nbOutputs) {
  ASSERT(nbInputs == 3);
  if (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
    return true;
  else return false;
}

void SelfAttention::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  ASSERT(nbInputs == 3);
  ctype = in[0].desc.type;
}

size_t SelfAttention::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

int SelfAttention::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
  const nvinfer1::PluginTensorDesc* outputDesc,
  const void* const* inputs, void* const* outputs,
  void* workspace, cudaStream_t stream) {
  int batch = inputDesc[0].dims.d[0];
  int seql = inputDesc[0].dims.d[1];
  if (seql > maxSize) {
    maxSize = seql + 256;
    CHECK(cudaFree(Score));
    CHECK(cudaFree(Score_));

    CHECK(cudaMalloc(&Score, batch*maxSize*maxSize*type2size(ctype)*n_Head));
    CHECK(cudaMalloc(&Score, batch*maxSize*maxSize*type2size(ctype)*n_Head));
  }

  CHECK(cudnnSetStream(dnnHandle, stream));
  CHECK(cublasSetStream(blasHandle, stream));
  // input q k v
  // q*k head = 1
  int h_feat = n_Feat / n_Head;
  if (ctype == DataType::kFLOAT) {
    float* Q = (float*)inputs[0];
    float* K = (float*)inputs[1];
    float* V = (float*)inputs[2];

    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[0][b + n * batch] = (float *)K + n * h_feat + b * seql * n_Feat;
        B[0][b + n * batch] = (float *)Q + n * h_feat + b * seql * n_Feat;
        C[0][b + n * batch] = (float *)Score + b * seql * seql + n * batch * seql * seql;
      }
    }
    CHECK(cudaMemcpyAsync(gA, A[0], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[0], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[0], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));

    float scalar[1] = { 1.0 / (sqrtf((float)h_feat)) };
    CHECK(cublasSgemmBatched(
      blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      seql, seql, h_feat,
      scalar,//halpha
      (float**)gA, n_Feat,
      (float**)gB, n_Feat, belta,
      (float**)gC, seql, batch * n_Head));

    if (Mask)
      MaskScore((float*)Score,batch,n_Feat,stream);
    CHECK(cudnnSetTensor4dDescriptor(xdes, CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, n_Head * batch * seql, seql, 1, 1));
    CHECK(cudnnSoftmaxForward(dnnHandle, algo_t, mode_t, alpha, xdes, Score, belta, xdes, Score_));

    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[1][b + n * batch] = (float *)V + n * h_feat + b * seql * n_Feat;
        B[1][b + n * batch] = (float *)Score_ + b * seql * seql + n * batch * seql * seql;
        C[1][b + n * batch] = (float *)outputs[0] + n * h_feat + b * seql * n_Feat;
      }
    }
    CHECK(cudaMemcpyAsync(gA, A[1], sizeof(void*)*batch * n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[1], sizeof(void*)*batch * n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[1], sizeof(void*)*batch * n_Head, cudaMemcpyHostToDevice, stream));

    CHECK(cublasSgemmBatched(
      blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      h_feat, seql, seql,
      alpha,
      (float**)gA, n_Feat,
      (float**)gB, seql,
      belta,
      (float**)gC, n_Feat, batch*n_Head));
  }
  else if (ctype == DataType::kHALF) {
    half* Q = (half*)inputs[0];
    half* K = (half*)inputs[1];
    half* V = (half*)inputs[2];

    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[0][b + n * batch] = (half *)K + n * h_feat + b * seql * n_Feat;
        B[0][b + n * batch] = (half *)Q + n * h_feat + b * seql * n_Feat;
        C[0][b + n * batch] = (half *)Score + b * seql * seql + n * batch* seql * seql;
      }
    }
    CHECK(cudaMemcpyAsync(gA, A[0], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[0], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[0], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));

    half scalar[1] = { 1.0 / (sqrtf((float)h_feat)) };
    CHECK(cublasHgemmBatched(
      blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      seql, seql, h_feat,
      scalar,//halpha,
      (half**)gA, n_Feat,
      (half**)gB, n_Feat, hbelta,
      (half**)gC, seql, batch * n_Head));

    if (Mask)
      MaskScore((half*)Score, batch, n_Feat, stream);
    CHECK(cudnnSetTensor4dDescriptor(xdes, CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_HALF, n_Head * batch * seql, seql, 1, 1));
    CHECK(cudnnSoftmaxForward(dnnHandle, algo_t, mode_t, alpha, xdes, Score, belta, xdes, Score_));

    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[1][b + n * batch] = (half *)V + n * h_feat + b * seql * n_Feat;
        B[1][b + n * batch] = (half *)Score_ + b * seql * seql + n * batch * seql * seql;
        C[1][b + n * batch] = (half *)outputs[0] + n * h_feat + b * seql * n_Feat;
      }
    }
    CHECK(cudaMemcpyAsync(gA, A[1], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[1], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[1], sizeof(void*)*batch*n_Head, cudaMemcpyHostToDevice, stream));

    CHECK(cublasHgemmBatched(
      blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      h_feat, seql, seql,
      halpha,
      (half**)gA, n_Feat,
      (half**)gB, seql,
      hbelta,
      (half**)gC, n_Feat, batch*n_Head));
  }

  return 0;
}

DataType SelfAttention::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return ctype;
}

const char* SelfAttention::getPluginVersion() const {
  return SELF_ATTENTION_PLUGIN_VERSION;
}
const char* SelfAttention::getPluginType() const {
  return SELF_ATTENTION_PLUGIN_NAME;
}

int SelfAttention::getNbOutputs() const {
  return 1;
}

int SelfAttention::initialize() {
  return 0;
}

void SelfAttention::terminate() {}

void SelfAttention::destroy() {}

void SelfAttention::serialize(void* buffer) const {
  char* ptr = (char*)buffer;
  write(ptr, n_Head);
  write(ptr, n_Feat);
  write(ptr, Mask);
  write(ptr, (int)ctype);
}

size_t SelfAttention::getSerializationSize() const {
  return sizeof(int) * 4;
}

const char* SelfAttention::getPluginNamespace() const {
  return mNameSpace.c_str();
}

void SelfAttention::setPluginNamespace(const char* pluginNamespace) {
  mNameSpace = pluginNamespace;
}

SelfAttentionPluginCreator::SelfAttentionPluginCreator() {

}
SelfAttentionPluginCreator::~SelfAttentionPluginCreator() {

}

const char* SelfAttentionPluginCreator::getPluginName() const {
  return SELF_ATTENTION_PLUGIN_NAME;
}

const char* SelfAttentionPluginCreator::getPluginVersion() const {
  return SELF_ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* SelfAttentionPluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* SelfAttentionPluginCreator::createPlugin(
  const char* name, const PluginFieldCollection* fc) {
  int nhead = ((int*)fc->fields[0].data)[0];
  int nfeat = ((int*)fc->fields[0].data)[1];
  return (IPluginV2DynamicExt*)new SelfAttention(nhead, nfeat);
}

IPluginV2DynamicExt* SelfAttentionPluginCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) {
  return (IPluginV2DynamicExt*)new SelfAttention(serialData, serialLength);
}