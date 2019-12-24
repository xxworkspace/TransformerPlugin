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
#include "srcAttentionPlugin.h"
#include <cstring>
#include <sstream>
#include <vector>

using namespace nvinfer1;
using nvinfer1::plugin::SrcAttention;
using nvinfer1::plugin::SrcAttentionPluginCreator;

namespace
{
  const char* SRC_ATTENTION_PLUGIN_VERSION{ "001" };
  const char* SRC_ATTENTION_PLUGIN_NAME{ "SrcAttention_TRT" };
} // namespace

PluginFieldCollection SrcAttentionPluginCreator::mFC{};
std::vector<PluginField> SrcAttentionPluginCreator::mPluginAttributes;

SrcAttention::SrcAttention(const Weights kweight, const Weights kbias,
  const Weights vweight, const Weights vbias,
  const int nhead, const int nfeat, DataType type){
  n_Head = nhead;
  n_Feat = nfeat;
  ctype = type;

  CHECK(cudaMalloc(&k_Weight, type2size(ctype)*n_Feat*n_Feat));
  CHECK(cudaMalloc(&k_Bias, type2size(ctype)*n_Feat));
  CHECK(cudaMalloc(&v_Weight, type2size(ctype)*n_Feat*n_Feat));
  CHECK(cudaMalloc(&v_Bias, type2size(ctype)*n_Feat));
  if (ctype == DataType::kHALF) {
      for (int i = 0; i < n_Feat*n_Feat; ++i) {
        ((half*)kweight.values)[i] = __float2half(((float*)kweight.values)[i]);
        ((half*)vweight.values)[i] = __float2half(((float*)vweight.values)[i]);
      }
      for (int i = 0; i < n_Feat; ++i) {
        ((half *)kbias.values)[i] = __float2half(((float*)kbias.values)[i]);
        ((half*)vbias.values)[i] = __float2half(((float*)vbias.values)[i]);
      }
  }

  CHECK(cudaMemcpy(k_Weight, kweight.values, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(k_Bias, kbias.values, type2size(ctype)*n_Feat, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(v_Weight, vweight.values, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
  CHECK(cudaMemcpy(v_Bias, vbias.values, type2size(ctype)*n_Feat, cudaMemcpyHostToDevice));

  //cudnn compute
  for (int n = 0; n < 2; ++n) {
    CHECK(cudaMallocHost(&A[n], sizeof(void *) * 1024));
    CHECK(cudaMallocHost(&B[n], sizeof(void *) * 1024));
    CHECK(cudaMallocHost(&C[n], sizeof(void *) * 1024));
  }
  CHECK(cudaMalloc(&gA, sizeof(void *) * 1024));
  CHECK(cudaMalloc(&gB, sizeof(void *) * 1024));
  CHECK(cudaMalloc(&gC, sizeof(void *) * 1024));


  CHECK(cudnnCreate(&dnnHandle));
  CHECK(cublasCreate(&blasHandle));

  CHECK(cudnnCreateTensorDescriptor(&kvdes));
  CHECK(cudnnCreateTensorDescriptor(&bdes));
  CHECK(cudnnCreateTensorDescriptor(&xdes));

  if (ctype == DataType::kFLOAT) {
    CHECK(cudnnSetTensor4dDescriptor(bdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_Feat, 1, 1));
  }
  else {
    CHECK(cudnnSetTensor4dDescriptor(bdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_HALF, 1, n_Feat, 1, 1));
  }
}

SrcAttention::SrcAttention(const SrcAttention& other) {
  ctype = other.ctype;
  n_Head = other.n_Head;
  n_Feat = other.n_Feat;

  CHECK(cudaMalloc(&k_Weight, type2size(ctype)*n_Feat*n_Feat));
  CHECK(cudaMalloc(&k_Bias, type2size(ctype)*n_Feat));
  CHECK(cudaMalloc(&v_Weight, type2size(ctype)*n_Feat*n_Feat));
  CHECK(cudaMalloc(&v_Bias, type2size(ctype)*n_Feat));

  CHECK(cudaMemcpy(k_Weight, other.k_Weight, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(k_Bias, other.k_Bias, type2size(ctype)*n_Feat, cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(v_Weight, other.v_Weight, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyDeviceToDevice));
  CHECK(cudaMemcpy(v_Bias, other.v_Bias, type2size(ctype)*n_Feat, cudaMemcpyDeviceToDevice));

  //cudnn compute
  for (int n = 0; n < 2; ++n) {
    CHECK(cudaMallocHost(&A[n], sizeof(void *) * 1024));
    CHECK(cudaMallocHost(&B[n], sizeof(void *) * 1024));
    CHECK(cudaMallocHost(&C[n], sizeof(void *) * 1024));
  }
  CHECK(cudaMalloc(&gA, sizeof(void *) * 1024));
  CHECK(cudaMalloc(&gB, sizeof(void *) * 1024));
  CHECK(cudaMalloc(&gC, sizeof(void *) * 1024));


  CHECK(cudnnCreate(&dnnHandle));
  CHECK(cublasCreate(&blasHandle));

  CHECK(cudnnCreateTensorDescriptor(&kvdes));
  CHECK(cudnnCreateTensorDescriptor(&bdes));
  CHECK(cudnnCreateTensorDescriptor(&xdes));

  if (ctype == DataType::kFLOAT) {
    CHECK(cudnnSetTensor4dDescriptor(bdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_Feat, 1, 1));
  }
  else {
    CHECK(cudnnSetTensor4dDescriptor(bdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_HALF, 1, n_Feat, 1, 1));
  }
}

SrcAttention::SrcAttention(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  n_Head = read<int>(ptr);
  n_Feat = read<int>(ptr);
  ctype = (DataType)read<int>(ptr);

  CHECK(cudaMalloc(&k_Weight, type2size(ctype)*n_Feat*n_Feat));
  CHECK(cudaMalloc(&k_Bias, type2size(ctype)*n_Feat));
  CHECK(cudaMalloc(&v_Weight, type2size(ctype)*n_Feat*n_Feat));
  CHECK(cudaMalloc(&v_Bias, type2size(ctype)*n_Feat));

  CHECK(cudaMemcpy(k_Weight, ptr, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
  ptr += type2size(ctype)*n_Feat*n_Feat;
  CHECK(cudaMemcpy(k_Bias, ptr, type2size(ctype)*n_Feat, cudaMemcpyHostToDevice));
  ptr += type2size(ctype)*n_Feat;
  CHECK(cudaMemcpy(v_Weight, ptr, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
  ptr += type2size(ctype)*n_Feat*n_Feat;
  CHECK(cudaMemcpy(v_Bias, ptr, type2size(ctype)*n_Feat, cudaMemcpyHostToDevice));

  //cudnn compute
  for (int n = 0; n < 2; ++n) {
    CHECK(cudaMallocHost(&A[n], sizeof(void *) * 1024));
    CHECK(cudaMallocHost(&B[n], sizeof(void *) * 1024));
    CHECK(cudaMallocHost(&C[n], sizeof(void *) * 1024));
  }
  CHECK(cudaMalloc(&gA, sizeof(void *) * 1024));
  CHECK(cudaMalloc(&gB, sizeof(void *) * 1024));
  CHECK(cudaMalloc(&gC, sizeof(void *) * 1024));

  CHECK(cudnnCreate(&dnnHandle));
  CHECK(cublasCreate(&blasHandle));

  CHECK(cudnnCreateTensorDescriptor(&kvdes));
  CHECK(cudnnCreateTensorDescriptor(&bdes));
  CHECK(cudnnCreateTensorDescriptor(&xdes));

  if (ctype == DataType::kFLOAT) {
    CHECK(cudnnSetTensor4dDescriptor(bdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_FLOAT, 1, n_Feat, 1, 1));
  }
  else {
    CHECK(cudnnSetTensor4dDescriptor(bdes, cudnnTensorFormat_t::CUDNN_TENSOR_NCHW, cudnnDataType_t::CUDNN_DATA_HALF, 1, n_Feat, 1, 1));
  }
}

IPluginV2DynamicExt* SrcAttention::clone()const {
  return (IPluginV2DynamicExt*)new SrcAttention(*this);
}

DimsExprs SrcAttention::getOutputDimensions(
  int outputIndex, const nvinfer1::DimsExprs* inputs,
  int nbInputs, nvinfer1::IExprBuilder& exprBuilder) {
  ASSERT(nbInputs == 1);
  return inputs[0];
}

bool SrcAttention::supportsFormatCombination(
  int pos, const nvinfer1::PluginTensorDesc* inOut,
  int nbInputs, int nbOutputs) {
  ASSERT(nbInputs == 1);
  if (inOut[0].type == DataType::kFLOAT || inOut[0].type == DataType::kHALF)
    return true;
  else return false;
}

void SrcAttention::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int nbInputs,
  const nvinfer1::DynamicPluginTensorDesc* out, int nbOutputs) {
  ASSERT(nbInputs == 1);
  ctype = in[0].desc.type;
}

size_t SrcAttention::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int nbInputs,
  const nvinfer1::PluginTensorDesc* outputs, int nbOutputs) const {
  return 0;
}

int SrcAttention::enqueue(const nvinfer1::PluginTensorDesc* inputDesc,
  const nvinfer1::PluginTensorDesc* outputDesc,
  const void* const* inputs, void* const* outputs,
  void* workspace, cudaStream_t stream) {
  int batch = inputDesc[0].dims.d[0];
  int seql = inputDesc[0].dims.d[1];
  int len = inputDesc[1].dims.d[1];

  if (seql > maxSeql || len > maxLen) {
    if (len > maxLen) {
      CHECK(cudaFree(K));
      CHECK(cudaFree(V));

      maxLen = len > maxLen ? len + 256 : maxLen;
      CHECK(cudaMalloc(&K, type2size(ctype) * batch * maxLen * maxSeql * n_Head));
      CHECK(cudaMalloc(&V, type2size(ctype) * batch * maxLen * maxSeql * n_Head));
    }

    CHECK(cudaFree(Score));
    CHECK(cudaFree(Score_));
    maxSeql = seql > maxSeql ? seql + 256 : maxSeql;
    CHECK(cudaMalloc(&Score, type2size(ctype) * batch * maxLen * maxSeql * n_Head));
    CHECK(cudaMalloc(&Score_, type2size(ctype) * batch * maxLen * maxSeql * n_Head));
  }

  CHECK(cublasSetStream(blasHandle, stream));
  CHECK(cudnnSetStream(dnnHandle, stream));
  if (ctype == DataType::kFLOAT) {
    float *Q = (float *)inputs[0];

    if (seql == 1) {
      CHECK(cudnnSetTensor4dDescriptor(kvdes, CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT, seql, n_Feat, 1, 1));
      CHECK(cublasSgemm(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n_Feat,
        seql, n_Feat, alpha, (float *)k_Weight,
        n_Feat, (float *)inputs[1], n_Feat, belta,
        (float *)K, n_Feat));
      CHECK(
        cudnnAddTensor(dnnHandle, alpha, bdes, k_Bias, alpha, kvdes, K));

      CHECK(cublasSgemm(blasHandle, CUBLAS_OP_T, CUBLAS_OP_N, n_Feat,
        seql, n_Feat, alpha, (float *)v_Weight,
        n_Feat, (float *)inputs[1], n_Feat, belta,
        (float *)V, n_Feat));
      CHECK(
        cudnnAddTensor(dnnHandle, alpha, bdes, v_Bias, alpha, kvdes, V));
    }
    // input q k v
    int h_feat = n_Feat / n_Head;
    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[0][b + n * batch] = (float *)K + n * h_feat;
        B[0][b + n * batch] = (float *)Q + n * h_feat + b * maxSeql * n_Feat;
        C[0][b + n * batch] = (float *)Score + b * seql * len + n * batch * seql * len;
      }
    }

    CHECK(cudaMemcpyAsync(gA, A[0], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[0], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[0], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));

    float scalar[1] = { 1.0 / (sqrtf((float)h_feat)) };
    CHECK(cublasSgemmBatched(
      blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      len, seql, h_feat,
      scalar,
      (float **)gA, n_Feat, (float **)gB, n_Feat,
      belta,
      (float **)gC, len,
      n_Head * batch));

    CHECK(
      cudnnSetTensor4dDescriptor(xdes, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT, n_Head * batch * seql, len, 1, 1));

    CHECK(cudnnSoftmaxForward(dnnHandle, algo_t, mode_t, alpha, xdes, Score, belta, xdes, Score_));

    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[1][b + n * batch] = (float *)V + n * h_feat;
        B[1][b + n * batch] = (float *)Score_ + b * seql * len + n * batch * seql * len;
        C[1][b + n * batch] = (float *)outputs[0] + n * h_feat + b * seql * n_Feat;
      }
    }

    CHECK(cudaMemcpyAsync(gA, A[1], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[1], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[1], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));

    CHECK(cublasSgemmBatched(
      blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      h_feat, seql, len,
      alpha, (float **)gA, n_Feat, (float **)gB, len,
      belta, (float **)gC, n_Feat,
      n_Head * batch));
  }
  else if (ctype == DataType::kHALF) {
    half* Q = (half*)inputs[0];

    if (len == 0) {
      CHECK(cudnnSetTensor4dDescriptor(kvdes, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, seql, n_Feat, 1, 1));
      CHECK(cublasHgemm(
        blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
        n_Feat, seql, n_Feat,
        halpha, (half *)k_Weight, n_Feat, (half *)inputs[1], n_Feat,
        hbelta, (half *)K, n_Feat));
      CHECK(
        cudnnAddTensor(dnnHandle, alpha, bdes, k_Bias, alpha, kvdes, K));

      CHECK(cublasHgemm(
        blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
        n_Feat, seql, n_Feat,
        halpha, (half *)v_Weight, n_Feat, (half *)inputs[1], n_Feat,
        hbelta, (half *)V, n_Feat));
      CHECK(
        cudnnAddTensor(dnnHandle, alpha, bdes, v_Bias, alpha, kvdes, V));
    }

    int h_feat = n_Feat / n_Head;
    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[0][b + n * batch] = (half *)K + n * h_feat;
        B[0][b + n * batch] = (half *)Q + n * h_feat + b * maxSeql * n_Feat;
        C[0][b + n * batch] = (half *)Score + b * seql * len + n * batch * seql * len;
      }
    }

    CHECK(cudaMemcpyAsync(gA, A[0], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[0], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[0], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));

    half scalar[1] = { 1.0 / (sqrtf((float)h_feat)) };
    CHECK(cublasHgemmBatched(
      blasHandle, CUBLAS_OP_T, CUBLAS_OP_N,
      len, seql, h_feat,
      scalar,//halpha,
      (half **)gA, n_Feat, (half **)gB, n_Feat,
      hbelta,
      (half **)gC, len,
      n_Head * batch));

    CHECK(cudnnSetTensor4dDescriptor(xdes, CUDNN_TENSOR_NCHW, CUDNN_DATA_HALF, n_Head * batch * seql, len, 1, 1));
    CHECK(cudnnSoftmaxForward(dnnHandle, algo_t, mode_t, alpha, xdes, Score,belta, xdes, Score_));

    for (int n = 0; n < n_Head; ++n) {
      for (int b = 0; b < batch; ++b) {
        A[1][b + n * batch] = (half *)V + n * h_feat;
        B[1][b + n * batch] = (half *)Score_ + b * seql * len + n * batch * seql * len;
        C[1][b + n * batch] = (half *)outputs[0] + n * h_feat + b * maxSeql * n_Feat;
      }
    }

    CHECK(cudaMemcpyAsync(gA, A[1], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gB, B[1], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));
    CHECK(cudaMemcpyAsync(gC, C[1], sizeof(void *) * n_Head * batch,
      cudaMemcpyHostToDevice, stream));

    CHECK(cublasHgemmBatched(
      blasHandle, CUBLAS_OP_N, CUBLAS_OP_N,
      h_feat, seql, len,
      halpha, (half **)gA, n_Feat, (half **)gB, len,
      hbelta, (half **)gC, n_Feat,
      n_Head * batch));
  }

  return 0;
}

DataType SrcAttention::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const {
  return ctype;
}

const char* SrcAttention::getPluginVersion() const {
  return SRC_ATTENTION_PLUGIN_VERSION;
}
const char* SrcAttention::getPluginType() const {
  return SRC_ATTENTION_PLUGIN_NAME;
}

int SrcAttention::getNbOutputs() const {
  return 1;
}

int SrcAttention::initialize() {
  return 0;
}

void SrcAttention::terminate() {}

void SrcAttention::destroy() {}

void SrcAttention::serialize(void* buffer) const {
  char* ptr = (char*)buffer;
  write(ptr, n_Head);
  write(ptr, n_Feat);
  write(ptr, (int)ctype);

  CHECK(cudaMemcpy(ptr, k_Weight, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyDeviceToHost));
  ptr += this->type2size(ctype)*n_Feat*n_Feat;
  CHECK(cudaMemcpy(ptr, k_Bias, type2size(ctype)*n_Feat, cudaMemcpyDeviceToHost));
  ptr += type2size(ctype)*n_Feat;
  CHECK(cudaMemcpy(ptr, v_Weight, type2size(ctype)*n_Feat*n_Feat, cudaMemcpyDeviceToHost));
  ptr += type2size(ctype)*n_Feat*n_Feat;
  CHECK(cudaMemcpy(ptr, v_Bias, type2size(ctype)*n_Feat, cudaMemcpyDeviceToHost));
}

size_t SrcAttention::getSerializationSize() const {
  size_t size = sizeof(int) * 3;
  size += (type2size(ctype)*n_Feat*n_Feat + type2size(ctype)*n_Feat) * 2;

  return size;
}

const char* SrcAttention::getPluginNamespace() const {
  return mNameSpace.c_str();
}

void SrcAttention::setPluginNamespace(const char* pluginNamespace) {
  mNameSpace = pluginNamespace;
}

SrcAttentionPluginCreator::SrcAttentionPluginCreator() {

}
SrcAttentionPluginCreator::~SrcAttentionPluginCreator() {

}

const char* SrcAttentionPluginCreator::getPluginName() const {
  return SRC_ATTENTION_PLUGIN_NAME;
}

const char* SrcAttentionPluginCreator::getPluginVersion() const {
  return SRC_ATTENTION_PLUGIN_VERSION;
}

const PluginFieldCollection* SrcAttentionPluginCreator::getFieldNames() {
  return &mFC;
}

IPluginV2DynamicExt* SrcAttentionPluginCreator::createPlugin(
  const char* name, const PluginFieldCollection* fc) {
  Weights kweight, kbias;
  Weights vweight, vbias;

  int nhead = ((int*)fc->fields[0].data)[0],
    nfeat = ((int*)fc->fields[0].data)[1];
  DataType ctype = (DataType)((int*)fc->fields[0].data)[2];

  kweight.type = DataType::kFLOAT;
  kweight.count = fc->fields[1].length;
  kweight.values = fc->fields[1].data;

  kbias.type = DataType::kFLOAT;
  kbias.count = fc->fields[2].length;
  kbias.values = fc->fields[2].data;

  vweight.type = DataType::kFLOAT;
  vweight.count = fc->fields[3].length;
  vweight.values = fc->fields[3].data;

  vbias.type = DataType::kFLOAT;
  vbias.count = fc->fields[4].length;
  vbias.values = fc->fields[4].data;

  return (IPluginV2DynamicExt*)new SrcAttention(kweight, kbias, vweight, vbias, nhead, nfeat, ctype);
}

IPluginV2DynamicExt* SrcAttentionPluginCreator::deserializePlugin(
  const char* name, const void* serialData, size_t serialLength) {
  return (IPluginV2DynamicExt*)new SrcAttention(serialData, serialLength);
}