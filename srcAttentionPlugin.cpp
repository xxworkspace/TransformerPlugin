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
#include "cublas_v2.h"
#include <cstring>
#include <cudnn.h>
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
  if (ctype == DataType::kFLOAT) {
    CHECK(cudaMalloc(&k_Weight, sizeof(float)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&k_Bias, sizeof(float)*n_Feat));
    CHECK(cudaMalloc(&v_Weight, sizeof(float)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&v_Bias, sizeof(float)*n_Feat));

    CHECK(cudaMemcpy(k_Weight, kweight.values, sizeof(float)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(k_Bias, kbias.values, sizeof(float)*n_Feat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v_Weight, vweight.values, sizeof(float)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v_Bias, vbias.values, sizeof(float)*n_Feat, cudaMemcpyHostToDevice));
  }
  else if (ctype == DataType::kHALF) {
    CHECK(cudaMalloc(&k_Weight, sizeof(half)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&k_Bias, sizeof(half)*n_Feat));
    CHECK(cudaMalloc(&v_Weight, sizeof(half)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&v_Bias, sizeof(half)*n_Feat));

    for (int i = 0; i < n_Feat*n_Feat; ++i) {
      ((half*)kweight.values)[i] = __float2half(((float*)kweight.values)[i]);
      ((half*)vweight.values)[i] = __float2half(((float*)vweight.values)[i]);
    }
    for (int i = 0; i < n_Feat; ++i) {
      ((half *)kbias.values)[i] = __float2half(((float*)kbias.values)[i]);
      ((half*)vbias.values)[i] = __float2half(((float*)vbias.values)[i]);
    }

    CHECK(cudaMemcpy(k_Weight, kweight.values, sizeof(half)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(k_Bias, kbias.values, sizeof(half)*n_Feat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v_Weight, vweight.values, sizeof(half)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(v_Bias, vbias.values, sizeof(half)*n_Feat, cudaMemcpyHostToDevice));
  }
}

SrcAttention::SrcAttention(const SrcAttention& other) {
  ctype = other.ctype;
  n_Head = other.n_Head;
  n_Feat = other.n_Feat;

  if (ctype == DataType::kFLOAT) {
    CHECK(cudaMalloc(&k_Weight, sizeof(float)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&k_Bias, sizeof(float)*n_Feat));
    CHECK(cudaMalloc(&v_Weight, sizeof(float)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&v_Bias, sizeof(float)*n_Feat));

    CHECK(cudaMemcpy(k_Weight, other.k_Weight, sizeof(float)*n_Feat*n_Feat, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(k_Bias, other.k_Bias, sizeof(float)*n_Feat, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(v_Weight, other.v_Weight, sizeof(float)*n_Feat*n_Feat, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(v_Bias, other.v_Bias, sizeof(float)*n_Feat, cudaMemcpyDeviceToDevice));
  }
  else {
    CHECK(cudaMalloc(&k_Weight, sizeof(half)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&k_Bias, sizeof(half)*n_Feat));

    CHECK(cudaMalloc(&v_Weight, sizeof(half)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&v_Bias, sizeof(half)*n_Feat));

    CHECK(cudaMemcpy(k_Weight, other.k_Weight, sizeof(half)*n_Feat*n_Feat, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(k_Bias, other.k_Bias, sizeof(half)*n_Feat, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(v_Weight, other.v_Weight, sizeof(half)*n_Feat*n_Feat, cudaMemcpyDeviceToDevice));
    CHECK(cudaMemcpy(v_Bias, other.v_Bias, sizeof(half)*n_Feat, cudaMemcpyDeviceToDevice));
  }
}

SrcAttention::SrcAttention(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  n_Head = read<int>(ptr);
  n_Feat = read<int>(ptr);
  ctype = (DataType)read<int>(ptr);

  if (ctype == DataType::kFLOAT) {
    CHECK(cudaMalloc(&k_Weight, sizeof(float)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&k_Bias, sizeof(float)*n_Feat));
    CHECK(cudaMalloc(&v_Weight, sizeof(float)*n_Feat*n_Feat));
    CHECK(cudaMalloc(&v_Bias, sizeof(float)*n_Feat));

    CHECK(cudaMemcpy(k_Weight, ptr, sizeof(float)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    ptr += sizeof(float)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(k_Bias, ptr, sizeof(float)*n_Feat, cudaMemcpyHostToDevice));
    ptr += sizeof(float)*n_Feat;
    CHECK(cudaMemcpy(v_Weight, ptr, sizeof(float)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    ptr += sizeof(float)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(v_Bias, ptr, sizeof(float)*n_Feat, cudaMemcpyHostToDevice));
  }
  else {
    CHECK(cudaMemcpy(k_Weight, ptr, sizeof(half)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    ptr += sizeof(half)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(k_Bias, ptr, sizeof(half)*n_Feat, cudaMemcpyHostToDevice));
    ptr += sizeof(half)*n_Feat;
    CHECK(cudaMemcpy(v_Weight, ptr, sizeof(half)*n_Feat*n_Feat, cudaMemcpyHostToDevice));
    ptr += sizeof(half)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(v_Bias, ptr, sizeof(half)*n_Feat, cudaMemcpyHostToDevice));
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

  if (ctype == DataType::kFLOAT) {
    CHECK(cudaMemcpy(ptr, k_Weight, sizeof(float)*n_Feat*n_Feat, cudaMemcpyDeviceToHost));
    ptr += sizeof(float)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(ptr, k_Bias, sizeof(float)*n_Feat, cudaMemcpyDeviceToHost));
    ptr += sizeof(float)*n_Feat;
    CHECK(cudaMemcpy(ptr, v_Weight, sizeof(float)*n_Feat*n_Feat, cudaMemcpyDeviceToHost));
    ptr += sizeof(float)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(ptr, v_Bias, sizeof(float)*n_Feat, cudaMemcpyDeviceToHost));
  }
  else {
    CHECK(cudaMemcpy(ptr, k_Weight, sizeof(half)*n_Feat*n_Feat, cudaMemcpyDeviceToHost));
    ptr += sizeof(half)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(ptr, k_Bias, sizeof(half)*n_Feat, cudaMemcpyDeviceToHost));
    ptr += sizeof(half)*n_Feat;
    CHECK(cudaMemcpy(ptr, v_Weight, sizeof(half)*n_Feat*n_Feat, cudaMemcpyDeviceToHost));
    ptr += sizeof(half)*n_Feat*n_Feat;
    CHECK(cudaMemcpy(ptr, v_Bias, sizeof(half)*n_Feat, cudaMemcpyDeviceToHost));
  }
}

size_t SrcAttention::getSerializationSize() const {
  size_t size = sizeof(int) * 3;
  if (ctype == DataType::kFLOAT)
    size += (sizeof(float)*n_Feat*n_Feat + sizeof(float)*n_Feat) * 2;
  else
    size += (sizeof(half)*n_Feat*n_Feat + sizeof(half)*n_Feat) * 2;

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