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

SelfAttention::SelfAttention(const int nhead,const int nfeat) {
  n_Head = nhead;
  n_Feat = nfeat;
}

SelfAttention::SelfAttention(const SelfAttention& other) {
  n_Head = other.n_Head;
  n_Feat = other.n_Feat;
  ctype = other.ctype;
  mNameSpace = other.mNameSpace;
}

SelfAttention::SelfAttention(const void* buffer, size_t length) {
  const char* ptr = (const char*)buffer;
  n_Head = read<int>(ptr);
  n_Feat = read<int>(ptr);
  ctype = (DataType)read<int>(ptr);
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
  int batchsize = inputDesc[0].dims.d[0];

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
  write(ptr, (int)ctype);
}

size_t SelfAttention::getSerializationSize() const {
  return sizeof(int) * 3;
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